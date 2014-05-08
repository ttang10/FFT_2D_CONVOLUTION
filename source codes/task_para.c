/* This is a program doing 2D convolution 
using Task and Data parallel model */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "mpi.h"

#include "fft.h"
#include "file_handler.h"

#define N 512


int main(int argc, char **argv)
{
    int i, j, k;
    double start, end;
    /* Time array */
    double time[9];
	double comm_time = 0;
	double comp_time = 0;
    int chunkSize;
    MPI_Status status;
    /* Being used in FFT */
    float data[N][N];
    /* Being used in mm */
    float input_1[N][N], input_2[N][N];
    /* Local matrix for FFT */
    float local_data[N][N];

    /* World rank and processor, related to MPI_COMM_WORLD */
    int world_id;
    int world_processor;

    /* Divided rank and processors for communication, related to taskcomm */
    int task_id;
    int task_processor;

    /* A complex array  storing the temp row to operate FFT */
    complex temp_data[N];

    /* Initialize rank and the number of processor for the MPI */
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_id);
    MPI_Comm_size(MPI_COMM_WORLD, &world_processor);

    /* Initialize a new vector for distributing columns */
    MPI_Datatype column, col;
    /* Column vector */
    MPI_Type_vector(N, 1, N, MPI_FLOAT, &col);
    MPI_Type_commit(&col);
    MPI_Type_create_resized(col, 0, 1*sizeof(float), &column);
    MPI_Type_commit(&column);

    int task = world_id%4;
    MPI_Comm taskcomm;
    /* Split the MPI_COMM_WORLD */
    MPI_Comm_split(MPI_COMM_WORLD, task, world_id, &taskcomm);
    MPI_Comm_rank(taskcomm, &task_id);
    MPI_Comm_size(taskcomm, &task_processor);

    /* Initialize inter communicators */
    MPI_Comm t1_t3_comm, t2_t3_comm, t3_t4_comm;

    /* Calculate chunkSize */
    chunkSize = N/task_processor;

    /* Get the start time of all program */
    if(world_id == 0){
        printf("2D convolution using MPI task and data parallelism\n");
        start = MPI_Wtime();
    }

    /* Each group completes work and send results by inter communicators */
    if(task == 0){
        // task 1
        /* Create an inter communicator for task 1 and task 3 */
        MPI_Intercomm_create(taskcomm, 0, MPI_COMM_WORLD, 2, 1, &t1_t3_comm);

        if(task_id == 0){
            time[0] = MPI_Wtime();

            /* Read file */
            readIm1File(data);
            time[1] = MPI_Wtime();

            printf("Group 1: Reading file 1_im1 takes %f s.\n", time[1] - time[0]);
        }

        /* Scatter data to local ranks */
        MPI_Scatter(data, chunkSize*N, MPI_FLOAT,
                    local_data, chunkSize*N, MPI_FLOAT,
                    0, taskcomm);

        /* Compute time for distributing data */
        if(task_id == 0){
            time[2] = MPI_Wtime();
            printf("Group 1: Scattering 1_im1(row) to each processor takes %f s.\n", time[2] - time[1]);
        }

        /* Do 1_im1 2d FFT */
        /* Row FFT */
        for(i = 0; i < chunkSize; i++){
            for(j = 0; j < N; j++){
                /* FFT each row for im1 */
                temp_data[j].r = local_data[i][j];
                temp_data[j].i = 0;
            }

            c_fft1d(temp_data, N, -1);

            for(j = 0; j < N; j++)
                local_data[i][j] = temp_data[j].r;
        }

        /* Gather all the data and distribute in columns */
        if(task_id == 0){
            time[3] = MPI_Wtime();
            printf("Group 1: FFT each row for 1_im1 takes %f s.\n", time[3] - time[2]);
        }

        /* Gather all the data of 1_im1 */
        MPI_Gather(local_data, chunkSize*N, MPI_FLOAT,
                    data, chunkSize*N, MPI_FLOAT,
                    0, taskcomm);

        if(task_id == 0){
            time[4] = MPI_Wtime();
            printf("Group 1: Gathering all the data of 1_im1(row) takes %f s.\n", time[4] - time[3]);
        }

        /* Scatter all the data to column local data */
        MPI_Scatter(data, chunkSize, column,
                    local_data, chunkSize, column,
                    0, taskcomm);

        if(task_id == 0){
            time[5] = MPI_Wtime();
            printf("Group 1: Scattering 1_im1(column) to each processor takes %f s.\n", time[5] - time[4]);
        }

        /* Column FFT */
        for(i = 0; i < chunkSize; i++){
            for(j = 0; j < N; j++){
                /* FFT each column for im1 */
                temp_data[j].r = local_data[j][i];
                temp_data[j].i = 0;
            }

            c_fft1d(temp_data, N, -1);

            for(j = 0; j < N; j++)
                local_data[j][i] = temp_data[j].r;
        }

        /* Gather all the columns from each rank */
        if(task_id == 0){
            time[6] = MPI_Wtime();
            printf("Group 1: FFT each column for 1_im1 takes %f s.\n", time[6] - time[5]);
        }

        MPI_Gather(local_data, chunkSize, column,
                    data, chunkSize, column,
                    0, taskcomm);

        /* Compute time and distribute data to do matrix multiplication */
        if(task_id == 0){
            time[7] = MPI_Wtime();
            printf("Group 1: Gathering all the data of 1_im1(column) takes %f s.\n", time[7] - time[6]);
            /* Total time */
            printf("Group 1: Total time for task 1 in group 1 takes %f s.\n", time[7] - time[0]);

			comm_time += time[7] - time[6] + time[5] - time[3] + time[2] - time[1];
			comp_time += time[6] - time[5] + time[3] - time[2];
            /* Send data to group 3 via the inter communicator */
            MPI_Send(data, N*N, MPI_FLOAT, task_id, 13, t1_t3_comm);
        }
    }
    else if(task == 1){
        // Task 2
        /* Create an inter communicator for task 2 and task 3 */
        MPI_Intercomm_create(taskcomm, 0, MPI_COMM_WORLD, 2, 2, &t2_t3_comm);

        if(task_id == 0){
            time[0] = MPI_Wtime();

            /* Read file */
            readIm2File(data);
            time[1] = MPI_Wtime();

            printf("Group 2: Reading file 1_im2 takes %f s.\n", time[1] - time[0]);
        }

        /* Scatter data to local ranks */
        MPI_Scatter(data, chunkSize*N, MPI_FLOAT,
                    local_data, chunkSize*N, MPI_FLOAT,
                    0, taskcomm);

        /* Compute time for distributing data */
        if(task_id == 0){
            time[2] = MPI_Wtime();
            printf("Group 2: Scatter 1_im2(row) to each processor takes %f s.\n", time[2] - time[1]);
        }

        /* Do 1_im1 2d FFT */
        /* Row FFT */
        for(i = 0; i < chunkSize; i++){
            for(j = 0; j < N; j++){
                /* FFT each row for im1 */
                temp_data[j].r = local_data[i][j];
                temp_data[j].i = 0;
            }

            c_fft1d(temp_data, N, -1);

            for(j = 0; j < N; j++)
                local_data[i][j] = temp_data[j].r;
        }

        /* Gather all the data and distribute in columns */
        if(task_id == 0){
            time[3] = MPI_Wtime();
            printf("Group 2: FFT each row for 1_im2 takes %f s.\n", time[3] - time[2]);
        }

        /* Gather all the data of 1_im1 */
        MPI_Gather(local_data, chunkSize*N, MPI_FLOAT,
                    data, chunkSize*N, MPI_FLOAT,
                    0, taskcomm);

        if(task_id == 0){
            time[4] = MPI_Wtime();
            printf("Group 2: Gather all the data of 1_im2(row) takes %f s.\n", time[4] - time[3]);
        }

        /* Scatter all the data to column local data */
        MPI_Scatter(data, chunkSize, column,
                    local_data, chunkSize, column,
                    0, taskcomm);

        if(task_id == 0){
            time[5] = MPI_Wtime();
            printf("Group 2: Scatter 1_im2(column) to each processor takes %f s.\n", time[5] - time[4]);
        }

        /* Column FFT */
        for(i = 0; i < chunkSize; i++){
            for(j = 0; j < N; j++){
                /* FFT each column for im1 */
                temp_data[j].r = local_data[j][i];
                temp_data[j].i = 0;
            }

            c_fft1d(temp_data, N, -1);

            for(j = 0; j < N; j++)
                local_data[j][i] = temp_data[j].r;
        }

        /* Gather all the columns from each rank */
        if(task_id == 0){
            time[6] = MPI_Wtime();
            printf("Group 2: FFT each column for 1_im2 takes %f s.\n", time[6] - time[5]);
        }

        MPI_Gather(local_data, chunkSize, column,
                    data, chunkSize, column,
                    0, taskcomm);

        /* Compute time and distribute data to do matrix multiplication */
        if(task_id == 0){
            time[7] = MPI_Wtime();
            printf("Group 2: Gather all the data of 1_im2(column) takes %f s.\n", time[7] - time[6]);
            /* Total time */
            printf("Group 2: Total time for task 2 in group 2 takes %f s.\n", time[7] - time[0]);
			
			comm_time += time[7] - time[6] + time[5] - time[3] + time[2] - time[1];
			comp_time += time[6] - time[5] + time[3] - time[2];
            /* Send data to group 3 via the inter communicator */
            MPI_Send(data, N*N, MPI_FLOAT, task_id, 23, t2_t3_comm);
        }
    }
    else if(task == 2){
        // Task 3
        /* Local matrix for matrix multiplication */
        float local_data2[chunkSize][N];
        /* Create inter communicators for task 1 and task3, task 2 and task 3, task 3 and task 4 */
        MPI_Intercomm_create(taskcomm, 0, MPI_COMM_WORLD, 0, 1, &t1_t3_comm);
        MPI_Intercomm_create(taskcomm, 0, MPI_COMM_WORLD, 1, 2, &t2_t3_comm);
        MPI_Intercomm_create(taskcomm, 0, MPI_COMM_WORLD, 3, 3, &t3_t4_comm);

        /* Receive data from group 1 and group 2 */
        if(task_id == 0){
            time[0] = MPI_Wtime();

            MPI_Recv(input_1, N*N, MPI_FLOAT, task_id, 13, t1_t3_comm, &status);
            MPI_Recv(input_2, N*N, MPI_FLOAT, task_id, 23, t2_t3_comm, &status);

            time[1] = MPI_Wtime();

            /* Time of receiving data from group 1 and group 2 */
            printf("Group 3: Receiving data from group 1 and group 2 takes %f s.\n", time[1] - time[0]);
        }

        /* Do matrix multiplication */
        MPI_Scatter(input_1, chunkSize*N, MPI_FLOAT,
                    local_data, chunkSize*N, MPI_FLOAT,
                    0, taskcomm);
        /* Broadcast data2 to all the ranks */
        MPI_Bcast(input_2, N*N, MPI_FLOAT, 0, taskcomm);

        if(task_id == 0){
            time[2] = MPI_Wtime();
            printf("Group 3: Scattering data for multiplication takes %f s.\n", time[2] - time[1]);
        }

        /* Matrix multiplication */
        for(i = 0; i < chunkSize; i++)
            for(j = 0; j < N; j++){
                local_data2[i][j] = 0;
                for(k = 0; k < N; k++)
                    local_data2[i][j] += local_data[i][k]*input_2[k][j];
            }

        /* Collect multiplication result from each rank */
        if(task_id == 0){
            time[3] = MPI_Wtime();
            printf("Group 3: Matrix multiplication takes %f s.\n", time[3] - time[2]);
        }

        /* Gather data */
        MPI_Gather(local_data2, chunkSize*N, MPI_FLOAT,
                   data, chunkSize*N, MPI_FLOAT,
                   0, taskcomm);

        if(task_id == 0){
            time[4] = MPI_Wtime();
            printf("Group 3: Gathering data after Matrix multiplication takes %f s.\n", time[4] - time[3]);
            /* total time */
            printf("Group 3: Total time for task 3 in group 3 takes %f s.\n", time[4] - time[0]);
            /* send result of matrix multiplication to group 4 */
            MPI_Send(data, N*N, MPI_FLOAT, task_id, 34, t3_t4_comm);
        }
		
		comm_time += time[4] - time[3] + time[2] - time[0];
		comp_time += time[3] - time[2];

        MPI_Comm_free(&t1_t3_comm);
        MPI_Comm_free(&t2_t3_comm);
    }
    else{
        // Task 4
        /* Create an inter communicator for task 3 and task 4 */
        MPI_Intercomm_create(taskcomm, 0, MPI_COMM_WORLD, 2, 3, &t3_t4_comm);

        /* Receive data from group 3 */
        if(task_id == 0){
            time[0] = MPI_Wtime();

            MPI_Recv(data, N*N, MPI_FLOAT, task_id, 34, t3_t4_comm, &status);

            time[1] = MPI_Wtime();
            printf("Group 4: Receiving data from group 3 takes %f s.\n", time[1] - time[0]);
        }

        /* Scatter data to each processor */
        MPI_Scatter(data, chunkSize*N, MPI_FLOAT,
                    local_data, chunkSize*N, MPI_FLOAT,
                    0, taskcomm);

        if(task_id == 0){
            time[2] = MPI_Wtime();
            printf("Group 4: Scattering data of rows to each processor takes %f s.\n", time[2] - time[1]);
        }

        /* Inverse-2DFFT(row) */
        for(i = 0; i < chunkSize; i++){
            for(j = 0; j < N; j++){
                /* FFT each row for im1 */
                temp_data[j].r = local_data[i][j];
                temp_data[j].i = 0;
            }

            c_fft1d(temp_data, N, 1);

            for(j = 0; j < N; j++)
                local_data[i][j] = temp_data[j].r;
        }

        if(task_id == 0){
            time[3] = MPI_Wtime();
            printf("Group 4: Inverse-2DFFT(row) takes %f s.\n", time[3] - time[2]);
        }
        /* Gather all the data */
        MPI_Gather(local_data, chunkSize*N, MPI_FLOAT,
                    data, chunkSize*N, MPI_FLOAT,
                    0, taskcomm);

        if(task_id == 0){
            time[4] = MPI_Wtime();
            printf("Group 4: Gathering data of Inverse-2DFFT(row) takes %f s.\n", time[4] - time[3]);
        }

        MPI_Scatter(data, chunkSize, column,
                    local_data, chunkSize, column,
                    0, taskcomm);

        if(task_id == 0){
            time[5] = MPI_Wtime();
            printf("Group 4: Scattering data of columns to each processor takes %f s.\n", time[5] - time[4]);
        }

        /* Inverse-2DFFT(column) for output file */
        for(i = 0; i < chunkSize; i++){
            for(j = 0; j < N; j++){
                /* FFT each column for im1 */
                temp_data[j].r = local_data[j][i];
                temp_data[j].i = 0;
            }

            c_fft1d(temp_data, N, 1);

            for(j = 0; j < N; j++)
                local_data[j][i] = temp_data[j].r;
        }

        if(task_id == 0){
            time[6] = MPI_Wtime();
            printf("Group 4: Inverse-2DFFT(column) takes %f s.\n", time[6] - time[5]);
        }

        /* Gather all the columns of output file from each rank */
        MPI_Gather(local_data, chunkSize, column,
                    data, chunkSize, column,
                    0, taskcomm);

        if(task_id == 0){
            time[7] = MPI_Wtime();
                printf("Group 4: Gathering data of Inverse-2DFFT(column) takes %f s.\n", time[7] - time[6]);

            writeFile(data);
            time[8] = MPI_Wtime();
            printf("Group 4: Writing file to out_1 takes %f s.\n", time[8] - time[7]);
			
			comm_time += time[7] - time[6] + time[5] - time[3] + time[2] - time[0];
			comp_time += time[6] - time[5] + time[3] - time[2];
        }
        MPI_Comm_free(&t3_t4_comm);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if(world_id == 0){
        end = MPI_Wtime();
		printf("Total communication time of 2D convolution using MPI task parallel takes %f s.\n", comm_time);
		printf("Total computing time of 2D convolution using MPI task parallel takes %f s.\n", comp_time);
		printf("Total running time without loading/writing of 2D convolution using MPI task parallel takes %f s.\n", comm_time + comp_time);
        printf("Total running time of 2D convolution using MPI task parallel takes %f s.\n", end - start);
    }

    /* Free vector and task comm */
    MPI_Type_free(&column);
    MPI_Type_free(&col);
    MPI_Comm_free(&taskcomm);
    MPI_Finalize();
    return 0;
}