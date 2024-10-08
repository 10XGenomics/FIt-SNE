#ifndef NBODYFFT_H
#define NBODYFFT_H

#include <mkl.h> 
#include "mkl_dfti.h"
#include <complex>

using namespace std;

typedef double (*kernel_type)(double, double, double);

typedef double (*kernel_type_2d)(double, double, double, double, double);

void create_dfti_handle(DFTI_DESCRIPTOR_HANDLE desc_handle, long long n_fft_coeffs, bool is_1_d);

void precompute_2d(double x_max, double x_min, double y_max, double y_min, int n_boxes, int n_interpolation_points,
                   kernel_type_2d kernel, double *box_lower_bounds, double *box_upper_bounds, double *y_tilde_spacings,
                   double *y_tilde, double *x_tilde, complex<double> *fft_kernel_tilde, double df);

void n_body_fft_2d(int N, int n_terms, double *xs, double *ys, double *chargesQij, int n_boxes,
                   int n_interpolation_points, double *box_lower_bounds, double *box_upper_bounds,
                   double *y_tilde_spacings, complex<double> *fft_kernel_tilde, double *potentialQij, unsigned int nthreads);

void precompute(double y_min, double y_max, int n_boxes, int n_interpolation_points, kernel_type kernel,
                double *box_lower_bounds, double *box_upper_bounds, double *y_tilde_spacing, double *y_tilde,
                complex<double> *fft_kernel_vector, double df);

void nbodyfft(int N, int n_terms, double *Y, double *chargesQij, int n_boxes, int n_interpolation_points,
              double *box_lower_bounds, double *box_upper_bounds, double *y_tilde_spacings, double *y_tilde,
              complex<double> *fft_kernel_vector, double *potentialsQij);

void interpolate(int n_interpolation_points, int N, const double *y_in_box, const double *y_tilde_spacings,
                 double *interpolated_values);

#endif
