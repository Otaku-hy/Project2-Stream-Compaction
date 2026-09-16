#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        static constexpr int blockSize = 64;

        /**
         * One level of the up-sweep (reduce) phase of the work-efficient scan.
         *
         * Only the threads that have work to do are launched: at level d there are
         * numThreads = paddedSize >> (d + 1) active nodes, and thread t owns the
         * node covering [t * stride, (t + 1) * stride), where stride = 1 << (d + 1).
         * Deriving the index from the compacted thread id (rather than testing
         * `index % stride == 0` over the whole array) keeps every launched warp busy.
         */
        __global__ void kernUpSweep(int numThreads, int stride, int *data) {
            int t = threadIdx.x + (blockIdx.x * blockDim.x);
            if (t >= numThreads) {
                return;
            }
            int right = t * stride + stride - 1;
            int left = right - (stride >> 1);
            data[right] += data[left];
        }

        /**
         * One level of the down-sweep phase of the work-efficient scan. Same
         * compacted indexing as the up-sweep.
         */
        __global__ void kernDownSweep(int numThreads, int stride, int *data) {
            int t = threadIdx.x + (blockIdx.x * blockDim.x);
            if (t >= numThreads) {
                return;
            }
            int right = t * stride + stride - 1;
            int left = right - (stride >> 1);
            int leftVal = data[left];
            data[left] = data[right];
            data[right] += leftVal;
        }

        /**
         * In-place exclusive scan over a device buffer of length paddedSize, which
         * must be a power of two with any padding beyond the real data zeroed.
         * Does no timing or allocation so that compact() can reuse it.
         */
        static void scanDevice(int paddedSize, int *dev_data) {
            int levels = ilog2ceil(paddedSize);

            // up-sweep
            for (int d = 0; d < levels; ++d) {
                int stride = 1 << (d + 1);
                int numThreads = paddedSize >> (d + 1);
                dim3 blocks((numThreads + blockSize - 1) / blockSize);
                kernUpSweep<<<blocks, blockSize>>>(numThreads, stride, dev_data);
            }

            // clear the last element (the total sum) to turn the reduction into a scan
            cudaMemset(dev_data + paddedSize - 1, 0, sizeof(int));

            // down-sweep
            for (int d = levels - 1; d >= 0; --d) {
                int stride = 1 << (d + 1);
                int numThreads = paddedSize >> (d + 1);
                dim3 blocks((numThreads + blockSize - 1) / blockSize);
                kernDownSweep<<<blocks, blockSize>>>(numThreads, stride, dev_data);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            // the binary tree needs a power-of-two sized array; pad the tail with 0s
            int paddedSize = 1 << ilog2ceil(n);

            int *dev_data;
            cudaMalloc((void**)&dev_data, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_data failed!");

            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_data failed!");
            if (paddedSize > n) {
                cudaMemset(dev_data + n, 0, (paddedSize - n) * sizeof(int));
            }

            timer().startGpuTimer();
            scanDevice(paddedSize, dev_data);
            timer().endGpuTimer();
            checkCUDAError("work-efficient scan kernels failed!");

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return 0;
            }

            int paddedSize = 1 << ilog2ceil(n);

            int *dev_idata, *dev_odata, *dev_bools, *dev_indices;
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_idata failed!");
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_odata failed!");
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bools failed!");
            cudaMalloc((void**)&dev_indices, paddedSize * sizeof(int));
            checkCUDAError("cudaMalloc dev_indices failed!");

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_idata failed!");

            dim3 fullBlocks((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            // map the input to 0s and 1s
            Common::kernMapToBoolean<<<fullBlocks, blockSize>>>(n, dev_bools, dev_idata);

            // exclusive scan of the flags gives the output index of each kept element
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            if (paddedSize > n) {
                cudaMemset(dev_indices + n, 0, (paddedSize - n) * sizeof(int));
            }
            scanDevice(paddedSize, dev_indices);

            // scatter the kept elements into their slots
            Common::kernScatter<<<fullBlocks, blockSize>>>(n, dev_odata, dev_idata, dev_bools, dev_indices);
            timer().endGpuTimer();
            checkCUDAError("work-efficient compact kernels failed!");

            // the count is the exclusive scan's last entry plus the last flag
            int lastIndex = 0, lastBool = 0;
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            int count = lastIndex + lastBool;

            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);
            return count;
        }
    }
}
