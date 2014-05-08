/* This is a program doing 2D convolution 
using SPMD model and using MPI collective 
communication functions wherever possible */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "mpi.h"

#include "fft.h"
#include "file_handler.h"

#define N 512

void convolution(int my_id, int p){
    int i, j, k;
    int chunkSize;
    double start, end;
    double time[14];

    /* Input data */
    float input_1[N][N], input_2[N][N];
    /* Output data */
    float output[N][N];
    /* Set the chunk size for each processor */
    chunkSize = N/p;

    /* Two arrays storing the local data distributed by rank 0 */
    float local_data1[N][N], local_data2[N][N];
    /* Local matrix for matrix multiplication */
    float local_data3[chunkSize][N];
    /* A complex array storing the temp row to operate FFT */
    complex temp_data[N];

    /* Initialization of the original Matrix and distribution of data */
    if(my_id == 0){
        printf("2D convolution using SPMD model and MPI Collective operations\n");
        start = MPI_Wtime();
        /*Read data from the files*/
        readFile(input_1, input_2);

        time[0] = MPI_Wtime();
        printf("Reading file takes %f s.\n", time[0] - start);
    }

    /* Scatter all the data to local data */
    MPI_Scatter(input_1, chunkSize*N, MPI_FLOAT,
                local_data1, chunkSize*N, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    MPI_Scatter(input_2, chunkSize*N, MPI_FLOAT,
                local_data2, chunkSize*N, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    /* Compute time for distributing data */
    if(my_id == 0){
        time[1] = MPI_Wtime();
        printf("Scattering data of rows to each processor takes %f s.\n", time[1] - time[0]);
    }

    /* Row FFT */
    for(i = 0; i < chunkSize; i++){
        for(j = 0; j < N; j++){
            /* FFT each row for im1 */
            temp_data[j].r = local_data1[i][j];
            temp_data[j].i = 0;
        }

        c_fft1d(temp_data, N, -1);

        for(j = 0; j < N; j++)
            local_data1[i][j] = temp_data[j].r;

        for(j = 0; j < N; j++){
            /* FFT each row for im2 */
            temp_data[j].r = local_data2[i][j];
            temp_data[j].i = 0;
        }

        c_fft1d(temp_data, N, -1);

        for(j = 0; j < N; j++)
            local_data2[i][j] = temp_data[j].r;
    }

    /* Gather all the data and distribute in columns */
    if(my_id == 0){
        time[2] = MPI_Wtime();
        printf("FFT each row for input im1 and im2 takes %f s.\n", time[2] - time[1]);
    }

    MPI_Gather(local_data1, chunkSize*N, MPI_FLOAT,
               input_1, chunkSize*N, MPI_FLOAT,
               0, MPI_COMM_WORLD);
    MPI_Gather(local_data2, chunkSize*N, MPI_FLOAT,
               input_2, chunkSize*N, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    if(my_id == 0){
        time[3] = MPI_Wtime();
        printf("Gathering all the data from different rows takes %f s.\n", time[3] - time[2]);
    }

    /* Initialize a new vector for distributing columns */
    MPI_Datatype column, col;
    /* Column vector */
    MPI_Type_vector(N, 1, N, MPI_FLOAT, &col);
    MPI_Type_commit(&col);
    MPI_Type_create_resized(col, 0, 1*sizeof(float), &column);
    MPI_Type_commit(&column);

    /* Scatter all the data to column local data */
    MPI_Scatter(input_1, chunkSize, column,
                local_data1, chunkSize, column,
                0, MPI_COMM_WORLD);
    MPI_Scatter(input_2, chunkSize, column,
                local_data2, chunkSize, column,
                0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if(my_id == 0){
        time[4] = MPI_Wtime();
        printf("Scattering data of columns to each processor takes %f s.\n", time[4] - time[3]);
    }
    /* Column FFT */
    for(i = 0; i < chunkSize; i++){
        for(j = 0; j < N; j++){
            /* FFT each column for im1 */
            temp_data[j].r = local_data1[j][i];
            temp_data[j].i = 0;
        }

        c_fft1d(temp_data, N, -1);

        for(j = 0; j < N; j++)
            local_data1[j][i] = temp_data[j].r;

        for(j = 0; j < N; j++){
            /* FFT each column for im2 */
            temp_data[j].r = local_data2[j][i];
            temp_data[j].i = 0;
        }

        c_fft1d(temp_data, N, -1);

        for(j = 0; j < N; j++)
            local_data2[j][i] = temp_data[j].r;
    }
    /* Gather all the columns from each rank */
    if(my_id == 0){
        time[5] = MPI_Wtime();
        printf("FFT each column for input im1 and im2 takes %f s.\n", time[5] - time[4]);
    }

    MPI_Gather(local_data1, chunkSize, column,
               input_1, chunkSize, column,
               0, MPI_COMM_WORLD);
    MPI_Gather(local_data2, chunkSize, column,
               input_2, chunkSize, column,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    /* Compute time and distribute data to do matrix multiplication */
    if(my_id == 0){
        time[6] = MPI_Wtime();
        printf("Gathering all the data from different columns takes %f s.\n", time[6] - time[5]);
    }

    MPI_Scatter(input_1, chunkSize*N, MPI_FLOAT,
                local_data1, chunkSize*N, MPI_FLOAT,
                0, MPI_COMM_WORLD);
    /* Broadcast data2 to all the ranks */
    MPI_Bcast(input_2, N*N, MPI_FLOAT, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if(my_id == 0){
        time[7] = MPI_Wtime();
        printf("Scattering data for multiplication takes %f s.\n", time[7] - time[6]);
    }

    /* Matrix multiplication */
    for(i = 0; i < chunkSize; i++)
        for(j = 0; j < N; j++)
            for(k = 0; k < N; k++)
                local_data3[i][j] += local_data1[i][k]*input_2[k][j];

    /* Collect multiplication results from each rank */
    if(my_id == 0){
        time[8] = MPI_Wtime();
        printf("Matrix multiplication takes %f s.\n", time[8] - time[7]);
    }

    /* Inverse-2DFFT(row) for the output file */
    for(i = 0; i < chunkSize; i++){
        for(j = 0; j < N; j++){
            /* FFT each row for im1 */
            temp_data[j].r = local_data3[i][j];
            temp_data[j].i = 0;
        }

        c_fft1d(temp_data, N, 1);

        for(j = 0; j < N; j++)
            local_data3[i][j] = temp_data[j].r;
    }

    if(my_id == 0){
        time[9] = MPI_Wtime();
        printf("Inverse-2DFFT for out_1(row) takes %f s.\n", time[9] - time[8]);
    }

    MPI_Gather(local_data3, chunkSize*N, MPI_FLOAT,
               output, chunkSize*N, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    if(my_id == 0){
        time[10] = MPI_Wtime();
        printf("Gathering all the data of Inverse-2DFFT for out_1(row) takes %f s.\n", time[10] - time[9]);
    }

    MPI_Scatter(output, chunkSize, column,
                local_data1, chunkSize, column,
                0, MPI_COMM_WORLD);

    if(my_id == 0){
        time[11] = MPI_Wtime();
        printf("Scattering out_1(column) to each processor takes %f s.\n", time[11] - time[10]);
    }

    /* Inverse-2DFFT(column) for the output file */
    for(i = 0; i < chunkSize; i++){
        for(j = 0; j < N; j++){
            /* FFT each column for im1 */
            temp_data[j].r = local_data1[j][i];
            temp_data[j].i = 0;
        }

        c_fft1d(temp_data, N, 1);

        for(j = 0; j < N; j++)
            local_data1[j][i] = temp_data[j].r;
    }

    /* Gathering all the columns of the output file from each rank */
    if(my_id == 0){
        time[12] = MPI_Wtime();
        printf("Inverse-2DFFT out_1(column) takes %f s.\n", time[12] - time[11]);
    }

    MPI_Gather(local_data1, chunkSize, column,
               output, chunkSize, column,
               0, MPI_COMM_WORLD);

    if(my_id == 0){
        time[13] = MPI_Wtime();
        printf("Gathering all the data of the output file(column) takes %f s.\n", time[13] - time[12]);

        writeFile(output);

        end = MPI_Wtime();
        printf("Writing the output file to file takes %f s.\n", end - time[13]);

        printf("Total communication time of 2D convolution using MPI_Scatter&MPI_Gather takes %f s.\n", time[13] - time[12] + time[11] - time[10] + time[7] - time[5] + time[4] - time[2] + time[1] - time[0]);
		printf("Total computing time of 2D convolution using MPI_Scatter&MPI_Gather takes %f s.\n", time[12] - time[11] + time[10] - time[7] + time[5] - time[4] + time[2] - time[1]);
		printf("Total running time without loading/writing of 2D convolution using MPI_Scatter&MPI_Gather takes %f s.\n", time[13] - time[0]);
		printf("Total running time of 2D convolution using MPI_Scatter&MPI_Gather takes %f s.\n", end - start);
    }

    /* Free vector column */
    MPI_Type_free(&column);
    MPI_Type_free(&col);
}


int main(int argc, char **argv)
{
    int my_id;
    int p;

    /* Initialize rank and number of processor for the MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_id);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    convolution(my_id, p);

    MPI_Finalize();
    return 0;
}
