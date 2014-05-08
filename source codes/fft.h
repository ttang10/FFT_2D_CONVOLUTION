#ifndef FFT_H
#define FFT_H

typedef struct {float r; float i;} complex;

extern void c_fft1d(complex *r, int n, int isign);

#endif // FFT_H
