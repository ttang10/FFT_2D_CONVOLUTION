#ifndef FILE_HANDLER_H
#define FILE_HANDLER_H

#define N 512

extern void readFile(float (*a)[N], float (*b)[N]);

extern void readIm1File(float (*a)[N]);

extern void readIm2File(float (*b)[N]);

extern void writeFile(float (*c)[N]);

#endif // FILE_HANDLER_H
