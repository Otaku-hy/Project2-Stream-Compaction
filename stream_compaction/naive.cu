#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        static constexpr int blockSize = 512;

        /**
         * One step of the naive (Hillis-Steele) inclusive scan:
         * odata[k] = idata[k] + idata[k - offset] for k >= offset, else idata[k].
         * Reads and writes must target different buffers to avoid races.
         */
        __global__ void kernNaiveScanStep(int n, int offset, int *odata, const int *idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            odata[k] = (k >= offset) ? idata[k - offset] + idata[k] : idata[k];
        }

        /**
         * Shifts an inclusive scan right by one to make it exclusive:
         * odata[0] = 0, odata[k] = idata[k - 1].
         */
        __global__ void kernInclusiveToExclusive(int n, int *odata, const int *idata) {
            int k = threadIdx.x + (blockIdx.x * blockDim.x);
            if (k >= n) {
                return;
            }
            odata[k] = (k == 0) ? 0 : idata[k - 1];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            if (n <= 0) {
                return;
            }

            int *dev_bufA, *dev_bufB;
            cudaMalloc((void**)&dev_bufA, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bufA failed!");
            cudaMalloc((void**)&dev_bufB, n * sizeof(int));
            checkCUDAError("cudaMalloc dev_bufB failed!");

            cudaMemcpy(dev_bufA, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAError("cudaMemcpy to dev_bufA failed!");

            dim3 fullBlocks((n + blockSize - 1) / blockSize);

            timer().startGpuTimer();
            for (int offset = 1; offset < n; offset <<= 1) {
                kernNaiveScanStep<<<fullBlocks, blockSize>>>(n, offset, dev_bufB, dev_bufA);
                std::swap(dev_bufA, dev_bufB);
            }
            // dev_bufA now holds the inclusive scan; shift to make it exclusive.
            kernInclusiveToExclusive<<<fullBlocks, blockSize>>>(n, dev_bufB, dev_bufA);
            timer().endGpuTimer();
            checkCUDAError("naive scan kernels failed!");

            cudaMemcpy(odata, dev_bufB, n * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAError("cudaMemcpy to odata failed!");

            cudaFree(dev_bufA);
            cudaFree(dev_bufB);
        }
    }
}
