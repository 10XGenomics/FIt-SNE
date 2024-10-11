#include <fstream>
#include <iostream>
#include <sstream> // For std::ostringstream
// Helper function for printing Y at each iteration. Useful for debugging
void print_progress(int iter, double *Y, int N, int no_dims) {

  std::ofstream myfile;
  std::ostringstream stringStream;
  stringStream << "dat/intermediate" << iter << ".txt";
  std::string copyOfStr = stringStream.str();
  myfile.open(stringStream.str().c_str());
  for (int j = 0; j < N; j++) {
    for (int i = 0; i < no_dims; i++) {
      myfile << Y[j * no_dims + i] << " ";
    }
    myfile << "\n";
  }
  myfile.close();
}
