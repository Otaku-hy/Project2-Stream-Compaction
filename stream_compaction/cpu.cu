#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Exclusive prefix sum, without touching the timer. Used internally so that
         * compactWithScan can reuse the scan logic without nesting timer calls.
         */
        static void scanImpl(int n, int *odata, const int *idata) {
            int sum = 0;
            for (int i = 0; i < n; ++i) {
                odata[i] = sum;
                sum += idata[i];
            }
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            scanImpl(n, odata, idata);
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[count++] = idata[i];
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            int *bools = new int[n];
            int *indices = new int[n];

            timer().startCpuTimer();
            // map to 0s and 1s
            for (int i = 0; i < n; ++i) {
                bools[i] = (idata[i] != 0) ? 1 : 0;
            }
            // exclusive scan of the boolean array gives each kept element its output index
            scanImpl(n, indices, bools);
            // scatter
            for (int i = 0; i < n; ++i) {
                if (bools[i]) {
                    odata[indices[i]] = idata[i];
                }
            }
            int count = (n > 0) ? (indices[n - 1] + bools[n - 1]) : 0;
            timer().endCpuTimer();

            delete[] bools;
            delete[] indices;
            return count;
        }
    }
}
