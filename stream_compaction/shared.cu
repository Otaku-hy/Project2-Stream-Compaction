#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "common.h"
#include "shared.h"

/**
 * Work-efficient scan with the tree levels inside the kernel (structure of GPU Gems 3,
 * Section 39.2.4; warp shuffles instead of the shared-memory tree of Example 39-2).
 *
 * Structure:
 *   1. kernBlockScan      - each block scans a tile of `tileSize` elements: 4 per thread in
 *                           registers, the per-thread totals across each warp with
 *                           __shfl_up_sync, and the warp totals across the block through
 *                           shared memory. Writes the exclusive scan of its tile and the
 *                           tile's total to blockSums[blockIdx].
 *   2. (recursion)        - blockSums is itself scanned with the same kernel, in place.
 *                           For n = 2^24 and a 1024-element tile: 16384 block sums -> 16 -> 1,
 *                           i.e. three block-scan launches and two add passes.
 *   3. kernAddBlockOffsets- each block adds the scanned block sum of the blocks before it
 *                           to every element of its tile.
 */

namespace StreamCompaction {
    namespace Shared {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        // Each thread owns four consecutive elements of the tile (one int4).
        static constexpr int blockSize = 256;
        static constexpr int tileSize = 4 * blockSize;
        static constexpr int numWarps = blockSize >> 5;

        /**
         * Inclusive scan across one warp (Kogge-Stone): x[i] += x[i - d] for i >= d.
         * Every lane executes every shuffle; only the add is guarded.
         */
        __device__ int warpScan(int val, int laneIdx) {
            int y = __shfl_up_sync(0xffffffff, val, 1);
            val += laneIdx < 1 ? 0 : y;
            y = __shfl_up_sync(0xffffffff, val, 2);
            val += laneIdx < 2 ? 0 : y;
            y = __shfl_up_sync(0xffffffff, val, 4);
            val += laneIdx < 4 ? 0 : y;
            y = __shfl_up_sync(0xffffffff, val, 8);
            val += laneIdx < 8 ? 0 : y;
            y = __shfl_up_sync(0xffffffff, val, 16);
            val += laneIdx < 16 ? 0 : y;
            return val;
        }

        /**
         * Exclusive scan of one tile. Elements past `n` are treated as 0 and never
         * written, so the last (partial) tile needs no host-side padding. If
         * blockSums != nullptr, the tile's total is written to blockSums[blockIdx.x].
         */
        __global__ void kernBlockScan(int n, int *odata, const int *idata, int *blockSums) {
            __shared__ int temp[blockSize];

            int warpIdx = threadIdx.x >> 5;
            int laneIdx = threadIdx.x & 31;
            int i = blockIdx.x * tileSize + threadIdx.x * 4;   // first of this thread's 4 elements

            // idata + i is 16-byte aligned: cudaMalloc base plus a multiple of 4 ints
            int4 iData4;
            if (i + 3 < n) {
                iData4 = *(const int4*)&idata[i];
            } else {
                iData4.x = (i     < n) ? idata[i]     : 0;
                iData4.y = (i + 1 < n) ? idata[i + 1] : 0;
                iData4.z = (i + 2 < n) ? idata[i + 2] : 0;
                iData4.w = (i + 3 < n) ? idata[i + 3] : 0;
            }

            // inclusive scan of the 4 elements in registers
            iData4.y += iData4.x;
            iData4.z += iData4.y;
            iData4.w += iData4.z;

            // inclusive scan of the per-thread totals across the warp
            int partialSum = warpScan(iData4.w, laneIdx);
            int warpSum = __shfl_sync(0xffffffff, partialSum, 31);

            if (laneIdx == 0) temp[warpIdx] = warpSum;
            __syncthreads();

            // warp 0 scans the warp totals; store the exclusive prefix and the tile total
            if (warpIdx == 0) {
                int own = threadIdx.x < numWarps ? temp[threadIdx.x] : 0;
                int incl = warpScan(own, laneIdx);
                temp[threadIdx.x] = incl - own;
                if (threadIdx.x == numWarps - 1 && blockSums) {
                    blockSums[blockIdx.x] = incl;
                }
            }
            __syncthreads();

            int warpPrefixSum = temp[warpIdx];
            partialSum += warpPrefixSum;

            //change to exclusive scan
            int exclusive = partialSum - iData4.w;
            int4 out = make_int4(exclusive, exclusive + iData4.x, exclusive + iData4.y, exclusive + iData4.z);
            if (i + 3 < n) {
                *(int4*)&odata[i] = out;
            } else {
                if (i     < n) odata[i]     = out.x;
                if (i + 1 < n) odata[i + 1] = out.y;
                if (i + 2 < n) odata[i + 2] = out.z;
                if (i + 3 < n) odata[i + 3] = out.w;
            }
        }

        /**
         * Adds offsets[blockIdx.x] (the exclusive scan of the block sums) to every
         * element of this block's tile.
         */
        __global__ void kernAddBlockOffsets(int n, int *data, const int *offsets) {
            int offset = offsets[blockIdx.x];
            int i = blockIdx.x * tileSize + threadIdx.x * 4;
            if (i + 3 < n) {
                int4 v = *(int4*)&data[i];
                v.x += offset; v.y += offset; v.z += offset; v.w += offset;
                *(int4*)&data[i] = v;
            } else {
                for (int k = 0; k < 4; ++k) {
                    if (i + k < n) data[i + k] += offset;
                }
            }
        }

        static int numBlocksFor(int n) {
            return (n + tileSize - 1) / tileSize;
        }

        /**
         * Allocates the chain of block-sum buffers needed to scan n elements:
         * one buffer per recursion level that has more than one block.
         * Done outside the timed region so timings only cover the kernels.
         */
        static std::vector<int*> allocBlockSums(int n) {
            std::vector<int*> sums;
            for (int m = numBlocksFor(n); m > 1; m = numBlocksFor(m)) {
                int *buf;
                cudaMalloc((void**)&buf, m * sizeof(int));
                checkCUDAError("cudaMalloc block sums failed!");
                sums.push_back(buf);
            }
            return sums;
        }

        static void freeBlockSums(std::vector<int*>& sums) {
            for (int *buf : sums) {
                cudaFree(buf);
            }
            sums.clear();
        }

        /**
         * Exclusive scan of n device ints from dev_in into dev_out (which may alias).
         * `sums[level]` is the block-sum buffer for this recursion level.
         */
        static void scanDevice(int n, int *dev_out, const int *dev_in,
                               const std::vector<int*>& sums, int level = 0) {
            int numBlocks = numBlocksFor(n);
            int *dev_sums = (numBlocks > 1) ? sums[level] : nullptr;

            kernBlockScan<<<numBlocks, blockSize>>>(n, dev_out, dev_in, dev_sums);

            if (numBlocks > 1) {
                // scan the block sums in place, then fold them back into each tile
                scanDevice(numBlocks, dev_sums, dev_sums, sums, level + 1);
                kernAddBlockOffsets<<<numBlocks, blockSize>>>(n, dev_out, dev_sums);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            int *dev_data;
            cudaMalloc((void**)&dev_data, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_data failed!");
            std::vector<int*> sums = allocBlockSums(n);

            timer().startGpuTimer();
            scanDevice(n, dev_data, dev_data, sums);
            timer().endGpuTimer();
            checkCUDAError("shared-memory scan kernels failed!");

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            freeBlockSums(sums);
            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return 0;
            }

            int *dev_idata, *dev_odata, *dev_bools, *dev_indices;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_indices, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_idata failed!");
            std::vector<int*> sums = allocBlockSums(n);

            // the map/scatter kernels use one element per thread
            constexpr int mapBlockSize = 256;
            dim3 mapBlocks((n + mapBlockSize - 1) / mapBlockSize);

            timer().startGpuTimer();
            Common::kernMapToBoolean<<<mapBlocks, mapBlockSize>>>(n, dev_bools, dev_idata);
            scanDevice(n, dev_indices, dev_bools, sums);
            Common::kernScatter<<<mapBlocks, mapBlockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();
            checkCUDAError("shared-memory compact kernels failed!");

            int lastIndex = 0, lastBool = 0;
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + lastBool;

            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            freeBlockSums(sums);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            return count;
        }
    }
}
