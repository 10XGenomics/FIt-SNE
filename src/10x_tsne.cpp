/* This file is a refactoring of tsne.cpp to have it use a stateful t-SNE.
   That leads to code duplication but is deliberate so one can easily compare
   the new (10x_tsne.cpp) to the original (tsne.cpp).  Basic change is we
   converted it to match the older  (stateful) API).  So instead of just the
   original `run` function, we divided that into an `init_tsne` and
   `step_tsne_by` function.

   The code was systematically changed to only produce output if `verbose` is
   specified as well. Error messages and exits were converted to `throw` No
   attempt was made to convert this to more modern C++.


*/

#include "10x_tsne.h"
#include "time_code.h"
#include <algorithm> // For std::max

// Same signature as TSNE::run with some things removed.
TSNEState::TSNEState(double *X, int N, int D, double *Y, int no_dims,
                     double perplexity, double theta, int rand_seed,
                     bool skip_random_init, int max_iter, int stop_lying_iter,
                     int mom_switch_iter, double momentum,
                     double final_momentum, double learning_rate, int K,
                     double sigma, int nbody_algorithm, int knn_algo,
                     double early_exag_coeff, bool no_momentum_during_exag,
                     int start_late_exag_iter, double late_exag_coeff,
                     int n_trees, int search_k, int nterms,
                     double intervals_per_integer, int min_num_intervals,
                     unsigned int nthreads, int load_affinities,
                     int perplexity_list_length, double *perplexity_list,
                     double df, double max_step_norm, bool verbose)
    : X(X), N(N), D(D), Y(Y), no_dims(no_dims), max_iter(max_iter), iter(0),
      stop_lying_iter(stop_lying_iter),
      start_late_exag_iter(start_late_exag_iter),
      mom_switch_iter(mom_switch_iter), early_exag_coeff(early_exag_coeff),
      late_exag_coeff(late_exag_coeff), n_trees(n_trees), search_k(search_k),
      momentum(momentum), final_momentum(final_momentum),
      learning_rate(learning_rate), K(K), sigma(sigma), theta(theta),
      max_step_norm(max_step_norm), nbody_algorithm(nbody_algorithm),
      knn_algo(knn_algo), measure_accuracy(false),
      no_momentum_during_exag(no_momentum_during_exag), df(df), nterms(nterms),
      load_affinities(load_affinities),
      intervals_per_integer(intervals_per_integer),
      min_num_intervals(min_num_intervals), nthreads(nthreads),
      perplexity(perplexity), perplexity_list(perplexity_list),
      perplexity_list_length(perplexity_list_length), itTest(0),
      verbose(verbose) {
  // Note throwing errors in constructors can lead to memory leaks, should check
  // this before calling it, this is a backup to that.
  if (knn_algo != 1 && knn_algo != 2) {
    throw std::invalid_argument("Invalid knn_algo param");
  }
  if (nbody_algorithm != 1 && nbody_algorithm != 2) {
    throw std::invalid_argument("n_body algorithm must be 1 or 2");
  }
  this->X = X;
  this->costs = (double *)calloc(max_iter, sizeof(double));
  if (costs == NULL) {
    throw std::runtime_error("Memory allocation failed!");
  }
  // Some logging messages
  if (N - 1 < 3 * perplexity) {
    throw std::invalid_argument(
        "Perplexity too large for the number of data points!");
  }
  if (verbose) {
    if (no_momentum_during_exag) {
      printf("No momentum during the exaggeration phase.\n");
    } else {
      printf("Will use momentum during exaggeration phase\n");
    }
  }
  // Used to call methods with
  TSNE *tsne = new TSNE(verbose);
  // Determine whether we are using an exact algorithm
  this->exact = theta == .0;

  // Allocate some memory
  this->dY = (double *)malloc(N * no_dims * sizeof(double));
  this->uY = (double *)malloc(N * no_dims * sizeof(double));
  this->gains = (double *)malloc(N * no_dims * sizeof(double));
  if (dY == nullptr || uY == nullptr || gains == nullptr)
    throw std::bad_alloc();
  // Initialize gradient to zeros and gains to ones.
  for (int i = 0; i < N * no_dims; i++)
    uY[i] = .0;
  for (int i = 0; i < N * no_dims; i++)
    gains[i] = 1.0;
  if (verbose) {
    printf("Computing input similarities...\n");
  }

  zeroMean(X, N, D);
  // if perplexity == 0, then what follows is the number of perplexities
  // to combine and then the list of these perpexities
  if (perplexity != 0) {
    perplexity_list_length = 0;
    perplexity_list = NULL;
  }
  if (perplexity > 0 || perplexity_list_length > 0) {
    if (verbose) {
      printf("Using perplexity, so normalizing input data (to prevent "
             "numerical problems)\n");
    }
    double max_X = .0;
    for (unsigned long i = 0; i < N * D; i++) {
      if (fabs(X[i]) > max_X)
        max_X = fabs(X[i]);
    }
    for (unsigned long i = 0; i < N * D; i++)
      X[i] /= max_X;
  } else if (verbose) {
    printf("Not using perplexity, so data are left un-normalized.\n");
  }
  // Compute input similarities for exact t-SNE
  this->P = nullptr;
  this->row_P = nullptr;
  this->col_P = nullptr;
  this->val_P = nullptr;
  if (exact) {
    // Compute similarities
    if (verbose) {
      printf("Theta set to 0, so running exact algorithm\n");
    }
    P = (double *)malloc(N * N * sizeof(double));
    if (P == NULL) {
      throw std::runtime_error("Memory allocation failed!");
    }

    tsne->computeGaussianPerplexity(X, N, D, P, perplexity, sigma,
                                    perplexity_list_length, perplexity_list);

    // Symmetrize input similarities
    if (verbose) {
      printf("Symmetrizing...\n");
    }
    int nN = 0;
    for (int n = 0; n < N; n++) {
      int mN = (n + 1) * N;
      for (int m = n + 1; m < N; m++) {
        P[nN + m] += P[mN + n];
        P[mN + n] = P[nN + m];
        mN += N;
      }
      nN += N;
    }
    double sum_P = .0;
    for (int i = 0; i < N * N; i++)
      sum_P += P[i];
    for (int i = 0; i < N * N; i++)
      P[i] /= sum_P;
    // sum_P is just a cute way of writing 2N
    if (verbose) {
      printf("Finished exact calculation of the P.  Sum_p: %lf \n", sum_P);
    }
  }
  // Compute input similarities for approximate t-SNE
  else {
    // Compute asymmetric pairwise input similarities
    int K_to_use;
    double sigma_to_use;

    if (perplexity < 0) {
      if (verbose) {
        printf("Using manually set kernel width\n");
      }
      K_to_use = K;
      sigma_to_use = sigma;
    } else {
      if (verbose) {
        printf("Using perplexity, not the manually set kernel width.  K "
               "(number of nearest neighbors) and sigma (bandwidth) parameters "
               "are going to be ignored.\n");
      }
      if (perplexity > 0) {
        K_to_use = (int)3 * perplexity;
      } else {
        K_to_use = (int)3 * perplexity_list[0];
        for (int pp = 1; pp < perplexity_list_length; pp++) {
          if ((int)3 * perplexity_list[pp] > K_to_use) {
            K_to_use = (int)3 * perplexity_list[pp];
          }
        }
      }
      sigma_to_use = -1;
    }

    if (knn_algo == 1) {
      if (verbose) {
        printf("Using ANNOY for knn search, with parameters: n_trees %d and "
               "search_k %d\n",
               n_trees, search_k);
      }
      int error_code = 0;
      error_code = tsne->computeGaussianPerplexity(
          X, N, D, &row_P, &col_P, &val_P, perplexity, K_to_use, sigma_to_use,
          n_trees, search_k, nthreads, perplexity_list_length, perplexity_list,
          rand_seed);
      if (error_code < 0)
        throw std::runtime_error(
            "Error code in Compute Gaussian Perplexity"); // error_code;
    } else if (knn_algo == 2) {
      if (verbose) {
        printf("Using VP trees for nearest neighbor search\n");
      }
      tsne->computeGaussianPerplexity(
          X, N, D, &row_P, &col_P, &val_P, perplexity, K_to_use, sigma_to_use,
          nthreads, perplexity_list_length, perplexity_list);
    }
    // Symmetrize input similarities
    if (verbose) {
      printf("Symmetrizing...\n");
    }
    tsne->symmetrizeMatrix(&row_P, &col_P, &val_P, N);
    double sum_P = .0;
    for (int i = 0; i < row_P[N]; i++)
      sum_P += val_P[i];
    for (int i = 0; i < row_P[N]; i++)
      val_P[i] /= sum_P;
  }

  // Set random seed
  if (skip_random_init != true) {
    if (rand_seed >= 0) {
      if (verbose) {
        printf("Using random seed: %d\n", rand_seed);
      }
      srand((unsigned int)rand_seed);
    } else {
      if (verbose) {
        printf("Using current time as random seed...\n");
      }
      srand(time(NULL));
    }
  }

  // Initialize solution (randomly)
  if (skip_random_init != true) {
    if (verbose) {
      printf("Randomly initializing the solution.\n");
    }
    for (int i = 0; i < N * no_dims; i++)
      Y[i] = tsne->randn() * .0001;
    if (verbose) {
      printf("Y[0] = %lf\n", Y[0]);
    }
  } else if (verbose) {
    printf("Using the given initialization.\n");
  }

  // If we are doing early exaggeration, we pre-multiply all the P by the
  // coefficient of early exaggeration
  double max_sum_cols = 0;
  // Compute maximum possible exaggeration coefficient, if user requests
  if (early_exag_coeff == 0) {
    for (int n = 0; n < N; n++) {
      double running_sum = 0;
      for (int i = row_P[n]; i < row_P[n + 1]; i++) {
        running_sum += val_P[i];
      }
      if (running_sum > max_sum_cols)
        max_sum_cols = running_sum;
    }
    early_exag_coeff = (1.0 / (learning_rate * max_sum_cols));
    if (verbose) {
      printf("Max of the val_Ps is: %lf\n", max_sum_cols);
    }
  }

  if (verbose) {
    printf("Exaggerating Ps by %f\n", early_exag_coeff);
  }
  if (exact) {
    for (int i = 0; i < N * N; i++) {
      P[i] *= early_exag_coeff;
    }
  } else {
    for (int i = 0; i < row_P[N]; i++)
      val_P[i] *= early_exag_coeff;
  }
  if (verbose) {
    print_progress(0, Y, N, no_dims);
  }
  // Perform main training loop
  if (exact & verbose) {
    printf("Input similarities computed \nLearning embedding...\n");
  } else if (verbose) {
    printf(
        "Input similarities computed (sparsity = %f)!\nLearning embedding...\n",
        (double)row_P[N] / ((double)N * (double)N));
  }

  std::chrono::steady_clock::time_point start_time =
      std::chrono::steady_clock::now();

  if (!exact) {
    if (nbody_algorithm == 2) {
      if (verbose) {
        printf("Using FIt-SNE approximation.\n");
      }
    } else if (nbody_algorithm == 1) {
      if (verbose) {
        printf("Using the Barnes-Hut approximation.\n");
      }
    }
  }
  delete (tsne);
}

void TSNEState::step_tsne_by(int step) {
  TSNE *tsne = new TSNE(verbose);
  const int cur_max = std::min(step + iter, max_iter);
  for (; iter < cur_max; iter++) {
    itTest = iter;
    if (exact) {
      // Compute the exact gradient using full P matrix
      tsne->computeExactGradient(P, Y, N, no_dims, dY, df);
    } else {
      if (nbody_algorithm == 2) {
        // Use FFT accelerated interpolation based negative gradients
        if (no_dims == 1) {
          if (df == 1.0) {
            tsne->computeFftGradientOneD(P, row_P, col_P, val_P, Y, N, no_dims,
                                         dY, nterms, intervals_per_integer,
                                         min_num_intervals, nthreads);
          } else {
            tsne->computeFftGradientOneDVariableDf(
                P, row_P, col_P, val_P, Y, N, no_dims, dY, nterms,
                intervals_per_integer, min_num_intervals, nthreads, df);
          }
        } else {
          if (df == 1.0) {
            tsne->computeFftGradient(P, row_P, col_P, val_P, Y, N, no_dims, dY,
                                     nterms, intervals_per_integer,
                                     min_num_intervals, nthreads);
          } else {
            tsne->computeFftGradientVariableDf(
                P, row_P, col_P, val_P, Y, N, no_dims, dY, nterms,
                intervals_per_integer, min_num_intervals, nthreads, df);
          }
        }
      } else if (nbody_algorithm == 1) {
        // Otherwise, compute the negative gradient using the Barnes-Hut
        // approximation
        tsne->computeGradient(P, row_P, col_P, val_P, Y, N, no_dims, dY, theta,
                              nthreads);
      }
    }
    // We can turn off momentum/gains until after the early exaggeration phase
    // is completed
    if (no_momentum_during_exag) {
      if (iter > stop_lying_iter) {
        for (int i = 0; i < N * no_dims; i++)
          gains[i] =
              (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
        for (int i = 0; i < N * no_dims; i++)
          if (gains[i] < .01)
            gains[i] = .01;
        for (int i = 0; i < N * no_dims; i++)
          uY[i] = momentum * uY[i] - learning_rate * gains[i] * dY[i];
        for (int i = 0; i < N * no_dims; i++)
          Y[i] = Y[i] + uY[i];
      } else {
        // During early exaggeration or compression, no trickery (i.e. no
        // momentum, or gains). Just good old fashion gradient descent
        for (int i = 0; i < N * no_dims; i++)
          Y[i] = Y[i] - dY[i];
      }
    } else {
      for (int i = 0; i < N * no_dims; i++)
        gains[i] =
            (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
      for (int i = 0; i < N * no_dims; i++)
        if (gains[i] < .01)
          gains[i] = .01;
      for (int i = 0; i < N * no_dims; i++)
        uY[i] = momentum * uY[i] - learning_rate * gains[i] * dY[i];

      // Clip the step sizes if max_step_norm is provided
      if (max_step_norm > 0) {
        for (int i = 0; i < N; i++) {
          double step = 0;
          for (int j = 0; j < no_dims; j++) {
            step += uY[i * no_dims + j] * uY[i * no_dims + j];
          }
          step = sqrt(step);
          if (step > max_step_norm) {
            for (int j = 0; j < no_dims; j++) {
              uY[i * no_dims + j] *= (max_step_norm / step);
            }
          }
        }
      }

      for (int i = 0; i < N * no_dims; i++)
        Y[i] = Y[i] + uY[i];
    }

    // Make solution zero-mean
    zeroMean(Y, N, no_dims);

    // Switch off early exaggeration
    if (iter == stop_lying_iter) {

      if (verbose) {
        printf("Unexaggerating Ps by %f\n", early_exag_coeff);
      }
      if (exact) {
        for (int i = 0; i < N * N; i++)
          P[i] /= early_exag_coeff;
      } else {
        for (int i = 0; i < row_P[N]; i++)
          val_P[i] /= early_exag_coeff;
      }
    }
    if (iter == start_late_exag_iter) {
      if (verbose) {
        printf("Exaggerating Ps by %f\n", late_exag_coeff);
      }
      if (exact) {
        for (int i = 0; i < N * N; i++)
          P[i] *= late_exag_coeff;
      } else {
        for (int i = 0; i < row_P[N]; i++)
          val_P[i] *= late_exag_coeff;
      }
    }
    if (iter == mom_switch_iter)
      momentum = final_momentum;

    // Print out progress
    if ((iter + 1) % 50 == 0 || iter == max_iter - 1) {
      INITIALIZE_TIME;
      START_TIME;
      // TODO: We have macros for timing and also a separate message, should use
      // only one approach.
      std::chrono::steady_clock::time_point start_time =
          std::chrono::steady_clock::now();
      double C = .0;
      if (exact) {
        C = tsne->evaluateError(P, Y, N, no_dims, df);
      } else {
        if (nbody_algorithm == 2) {
          C = tsne->evaluateErrorFft(row_P, col_P, val_P, Y, N, no_dims,
                                     nthreads, df);
        } else {
          C = tsne->evaluateError(row_P, col_P, val_P, Y, N, no_dims, theta,
                                  nthreads);
        }
      }

      // Adjusting the KL divergence if exaggeration is currently turned on
      // See
      // https://github.com/pavlin-policar/fastTSNE/blob/master/notes/notes.pdf,
      // Section 3.2
      if (iter < stop_lying_iter && stop_lying_iter != -1) {
        C = C / early_exag_coeff - log(early_exag_coeff);
      }
      if (iter >= start_late_exag_iter && start_late_exag_iter != -1) {
        C = C / late_exag_coeff - log(late_exag_coeff);
      }

      costs[iter] = C;
      END_TIME("Computing Error");

      std::chrono::steady_clock::time_point now =
          std::chrono::steady_clock::now();
      if (verbose) {
        printf("Iteration %d (50 iterations in %.2f seconds), cost %f\n",
               iter + 1,
               std::chrono::duration_cast<std::chrono::milliseconds>(now -
                                                                     start_time)
                       .count() /
                   (float)1000.0,
               C);
      }
      start_time = std::chrono::steady_clock::now();
    }
    delete (tsne);
  }
}

TSNEState::~TSNEState() {
  free(dY);
  free(uY);
  free(gains);
  free(costs);
  if (P != NULL) {
    free(P);
  }
  if (col_P != nullptr) {
    free(col_P);
  }
  if (row_P != nullptr) {
    free(row_P);
  }
  if (val_P != nullptr) {
    free(val_P);
  }
  // I've disabled the use of this for now, left enough bits in to make it easy
  // to resurect.
  // if(perplexity_list != NULL) {free(perplexity_list);}
}
