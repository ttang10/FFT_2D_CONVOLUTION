#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "file_handler.h"

#define N 512

void readFile(float (*a)[N], float (*b)[N]){
    int i, j;
    FILE *f1, *f2;

    f1 = fopen("1_im1", "r");
    f2 = fopen("1_im2", "r");

    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++){
            fscanf(f1, "%g", &a[i][j]);
            fscanf(f2, "%g", &b[i][j]);
        }

    fclose(f1);
    fclose(f2);
}

void readIm1File(float (*a)[N]){
    int i, j;
    FILE *f1;

    f1 = fopen("1_im1", "r");

    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++)
            fscanf(f1, "%g", &a[i][j]);

    fclose(f1);
}

void readIm2File(float (*b)[N]){
    int i, j;
    FILE *f2;

    f2 = fopen("1_im2", "r");

    for(i = 0; i < N; i++)
        for(j = 0; j < N; j++)
            fscanf(f2, "%g", &b[i][j]);

    fclose(f2);
}

void writeFile(float (*c)[N]){
    int i, j;
    FILE *f3;

    f3 = fopen("out_1", "w");

    for(i = 0; i < N; i++){
        for(j = 0; j < N; j++)
            fprintf(f3, "%.7e", c[i][j]);
        fprintf(f3, "\n");
    }

    fclose(f3);
}
