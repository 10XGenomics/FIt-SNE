// Modified version of the original API to match the old BH t-SNE one we used to
// use as much as possible.  The FIt-SNE code was written as a stand alone
// executable that would load data from files and process it all in one go.  We
// refactored this to make it a class that had an initialization followed by
// calls to run steps of the algorithm.  10X_tsne.cpp is the result of
// refactoring tsne.cpp into this format.

#include "tsne.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

// V2 to indicate we went from BH to FIt-SNE
struct TSNE_V2;

class TSNEState {
public:
  // Data and settings
  const double *X;
  const int N; // Number of points
  const int D;
  const int no_dims;          // Number of dimensions (e.g., 2 or 3)
  int iter;                   // keep track of how many iterations we've done.
  const int max_iter;         // Maximum number of iterations
  int stop_lying_iter;        // Iteration to stop early exaggeration
  int start_late_exag_iter;   // Iteration to start late exaggeration
  int mom_switch_iter;        // Iteration to switch momentum
  double early_exag_coeff;    // Coefficient for early exaggeration
  double late_exag_coeff;     // Coefficient for late exaggeration
  double momentum;            // Initial momentum
  double final_momentum;      // Final momentum
  const double learning_rate; // Learning rate for gradient descent
  const int K;
  const double theta;   // Barnes-Hut parameter
  double max_step_norm; // Maximum norm of gradient step
  int nbody_algorithm;  // Algorithm for n-body calculation
  int n_trees;
  int search_k;
  int knn_algo;
  bool exact;
  int load_affinities;

  bool measure_accuracy;        // Flag to measure accuracy during optimization
  bool no_momentum_during_exag; // Flag to disable momentum during exaggeration
  double df;                    // Degree of freedom or related parameter
  bool verbose;
  double perplexity;
  double *perplexity_list;
  const int perplexity_list_length;
  double sigma;
  // Working variables
  double *P; // Probability matrix

  unsigned int *row_P; // Row indices of sparse matrix
  unsigned int *col_P; // Column indices of sparse matrix
  double *val_P;       // Values of sparse matrix
  double *Y;           // Solution (positions in low-dimensional space)

  double *dY;    // Gradient
  double *uY;    // Velocity for gradient descent
  double *gains; // Per-dimension step size gains
  double *costs; // Costs over iterations

  int nterms;                   // FFT related term count
  double intervals_per_integer; // FFT parameter
  int min_num_intervals;        // FFT parameter
  int nthreads;                 // Number of threads
  int itTest;
  TSNEState(double *X, int N, int D, double *Y, int no_dims, double perplexity,
            double theta, int rand_seed, bool skip_random_init, int max_iter,
            int stop_lying_iter, int mom_switch_iter, double momentum,
            double final_momentum, double learning_rate, int K, double sigma,
            int nbody_algorithm, int knn_algo, double early_exag_coeff,
            bool no_momentum_during_exag, int start_late_exag_iter,
            double late_exag_coeff, int n_trees, int search_k, int nterms,
            double intervals_per_integer, int min_num_intervals,
            unsigned int nthreads, int load_affinities,
            int perplexity_list_length, double *perplexity_list, double df,
            double max_step_norm, bool verbose);
  ~TSNEState();
  void step_tsne_by(int step);
};

extern "C" {
TSNE_V2 *init_tsne(double *X, int N, int D, double *Y, int no_dims,
                   double perplexity, double theta, int rand_seed,
                   bool skip_random_init, const double *init, bool use_init,
                   int max_iter, int stop_lying_iter, int mom_switch_iter,
                   // New interface in FIt-SNE
                   double momentum, double learning_rate, double sigma,
                   int start_late_exag_iter, double late_exag_coeff,
                   int nbody_algorithm, int knn_algo, int search_k, int nterms,
                   double max_step_norm, unsigned int nthreads, bool verbose) {

  // To keep things simple, don't expose all the API quite yet.
  double final_momentum = 0.8;
  int K = -1;
  double early_exag_coeff = 12.0;
  bool no_momentum_during_exag = false;
  int n_trees = 50;
  int min_num_intervals = 50;
  double intervals_per_integer = 1.0;
  double *perplexity_list = NULL;
  int perplexity_list_length = 0;
  int load_affinities = 0;
  double df = 1.0;

  TSNEState *cur_tsne = new TSNEState(
      X, N, D, Y, no_dims, perplexity, theta, rand_seed, skip_random_init,
      max_iter, stop_lying_iter, mom_switch_iter,
      // Below are new to this interface
      momentum, final_momentum, learning_rate, K, sigma, nbody_algorithm,
      knn_algo, early_exag_coeff, no_momentum_during_exag, start_late_exag_iter,
      late_exag_coeff, n_trees, search_k, nterms, intervals_per_integer,
      min_num_intervals, nthreads, load_affinities, perplexity_list_length,
      perplexity_list, df, max_step_norm, verbose);
  return reinterpret_cast<TSNE_V2 *>(cur_tsne);
}

bool step_tsne_by(TSNE_V2 *tsne, int step) {
  auto cur_tsne = reinterpret_cast<TSNEState *>(tsne);
  cur_tsne->step_tsne_by(step);
  return true;
}
void free_tsne(TSNE_V2 *tsne) {
  auto cur_tsne = reinterpret_cast<TSNEState *>(tsne);
  delete cur_tsne;
}
}