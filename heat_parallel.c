// heat_parallel.c
#include <mpi.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define NX 500
#define NY 500
#define MAX_ITER 1000
#define TOLERANCE 1e-6

// Convert (local_row, col) to 1D index in a (local_rows+2) x NY array
static inline int idx(int row, int col) {
    return row * NY + col;
}

// Initialize local subdomain (rows 1..local_rows; 0 and local_rows+1 are halos)
static void initialize_subdomain(double *u, int local_rows, int start_row) {
    for (int i = 1; i <= local_rows; i++) {
        int global_row = start_row + i - 1;
        for (int j = 0; j < NY; j++) {
            double value = 0.0;
            if (global_row == 0 || global_row == NX - 1 || j == 0 || j == NY - 1) {
                value = 100.0;  // Boundary conditions
            }
            u[idx(i, j)] = value;
        }
    }
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 1D row-wise decomposition with load balancing
    int base_rows = NX / size;
    int remainder = NX % size;
    int local_rows = base_rows + (rank < remainder ? 1 : 0);
    int start_row = rank * base_rows + (rank < remainder ? rank : remainder);

    // Allocate with two halo rows: [0] and [local_rows+1]
    double *u     = calloc((local_rows + 2) * NY, sizeof(double));
    double *u_new = calloc((local_rows + 2) * NY, sizeof(double));
    if (!u || !u_new) {
        fprintf(stderr, "Rank %d failed to allocate memory\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    initialize_subdomain(u, local_rows, start_row);

    double start_time = MPI_Wtime();
    double global_diff = 0.0;

    for (int iter = 0; iter < MAX_ITER; iter++) {
        MPI_Request reqs[4];
        int req_count = 0;

        // Exchange boundary rows with neighbors (halo update)
        if (rank > 0) {
            // Receive from rank-1 into top halo row 0
            MPI_Irecv(&u[idx(0, 0)], NY, MPI_DOUBLE, rank - 1, 0,
                      MPI_COMM_WORLD, &reqs[req_count++]);
        }
        if (rank < size - 1) {
            // Receive from rank+1 into bottom halo row local_rows+1
            MPI_Irecv(&u[idx(local_rows + 1, 0)], NY, MPI_DOUBLE, rank + 1, 1,
                      MPI_COMM_WORLD, &reqs[req_count++]);
        }
        if (rank > 0) {
            // Send first interior row (1) to rank-1
            MPI_Isend(&u[idx(1, 0)], NY, MPI_DOUBLE, rank - 1, 1,
                      MPI_COMM_WORLD, &reqs[req_count++]);
        }
        if (rank < size - 1) {
            // Send last interior row (local_rows) to rank+1
            MPI_Isend(&u[idx(local_rows, 0)], NY, MPI_DOUBLE, rank + 1, 0,
                      MPI_COMM_WORLD, &reqs[req_count++]);
        }

        MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

        double local_max_diff = 0.0;

        // Jacobi update on interior points (skip global boundary rows)
        #pragma omp parallel for reduction(max:local_max_diff) schedule(static)
        for (int i = 1; i <= local_rows; i++) {
            int global_row = start_row + i - 1;
            if (global_row == 0 || global_row == NX - 1) {
                continue;  // boundary rows stay fixed
            }

            for (int j = 1; j < NY - 1; j++) {
                u_new[idx(i, j)] = 0.25 * (u[idx(i + 1, j)] + u[idx(i - 1, j)] +
                                           u[idx(i, j + 1)] + u[idx(i, j - 1)]);
                double diff = fabs(u_new[idx(i, j)] - u[idx(i, j)]);
                if (diff > local_max_diff) {
                    local_max_diff = diff;
                }
            }
        }

        // Update u with u_new (again skipping global boundaries)
        #pragma omp parallel for schedule(static)
        for (int i = 1; i <= local_rows; i++) {
            int global_row = start_row + i - 1;
            if (global_row == 0 || global_row == NX - 1) {
                continue;
            }
            for (int j = 1; j < NY - 1; j++) {
                u[idx(i, j)] = u_new[idx(i, j)];
            }
        }

        // Global max diff to check convergence
        MPI_Allreduce(&local_max_diff, &global_diff, 1, MPI_DOUBLE,
                      MPI_MAX, MPI_COMM_WORLD);

        if (global_diff < TOLERANCE) {
            if (rank == 0) {
                printf("Parallel (MPI+OpenMP) converged after %d iterations.\n", iter);
            }
            break;  // all ranks break together
        }
    }

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    double max_time = 0.0;
    MPI_Reduce(&elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Max wall time across ranks: %f seconds\n", max_time);
    }

    free(u);
    free(u_new);
    MPI_Finalize();
    return 0;
}
