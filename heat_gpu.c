#include <math.h>
#include <omp.h>
#include <stdio.h>

#define NX 500
#define NY 500
#define MAX_ITER 1000
#define TOLERANCE 1e-6

static double u[NX][NY];
static double u_new[NX][NY];

int main(void) {
    /* Initialize */
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            u[i][j] = 0.0;
            if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1) {
                u[i][j] = 100.0;
            }
        }
    }

    double start = omp_get_wtime();
    double max_diff = 0.0;

    #pragma acc data copy(u) create(u_new)
    for (int iter = 0; iter < MAX_ITER; iter++) {
        max_diff = 0.0;

        #pragma acc parallel loop collapse(2) reduction(max:max_diff)
        for (int i = 1; i < NX - 1; i++) {
            for (int j = 1; j < NY - 1; j++) {
                u_new[i][j] = 0.25 * (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1]);
                double diff = fabs(u_new[i][j] - u[i][j]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
            }
        }

        #pragma acc parallel loop collapse(2)
        for (int i = 1; i < NX - 1; i++) {
            for (int j = 1; j < NY - 1; j++) {
                u[i][j] = u_new[i][j];
            }
        }

        if (max_diff < TOLERANCE) {
            printf("Converged after %d iterations.\n", iter);
            break;
        }
    }

    double end = omp_get_wtime();
    printf("GPU-accelerated elapsed time: %f seconds\n", end - start);
    return 0;
}
