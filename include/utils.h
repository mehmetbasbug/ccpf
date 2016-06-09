#ifndef UTILS_H_
#define UTILS_H_
#include <algorithm>
#include "boost/multi_array.hpp"
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/expm1.hpp>
#include <boost/property_tree/ptree.hpp>
#include <cmath>
#include <cstdlib>
#include "H5Cpp.h"
#include <limits>
#include <map>
#include <vector>
using namespace std;
namespace bptree = boost::property_tree;

struct missing_entry {
  unsigned long int user, item;
};
struct entry {
  unsigned long int user, item;
  double response;
};
struct prediction {
  int label;
  double pred;
  bool operator<(const prediction& a) const { return pred < a.pred; }
};
struct test_result {
  double pred;
  double test_loglik;
  double test_error;
  double cond_pred;
  double cond_test_loglik;
  double cond_test_error;
};
struct normal_params {
  double mu, sigma2;
};
struct gamma_params {
  double shape, rate;
};
struct inverse_gaussian_params {
  double mu, lambda;
};
struct poisson_params {
  double lambda;
};
struct binomial_params {
  int r;
  double p;
};
struct negative_binomial_params {
  double r, p;
};
struct ztp_params {
  double lambda;
};
typedef boost::multi_array<double, 2> array2d;
typedef array2d::index index2d;
typedef vector<missing_entry> tuples;
typedef vector<entry> triplets;
typedef vector<prediction> predictions;
typedef map<unsigned long int, double> itemmap;
typedef vector<itemmap> itemmaps;
double trigamma(double);
double lambertw(const double);
double logsumexp(double*, size_t);
double logsumexp(vector<double>&, size_t);
double logsumexp2(double*, size_t);
double logsumexp2(vector<double>&, size_t);
normal_params mle_normal(const triplets&, unsigned long int);
gamma_params mle_gamma(const triplets&, unsigned long int);
inverse_gaussian_params mle_inverse_gaussian(const triplets&,
                                             unsigned long int);
poisson_params mle_poisson(const triplets&, unsigned long int);
double binomial_loglik(double, itemmap&, double);
binomial_params mle_binomial(const triplets&, unsigned long int);
double negative_binomial_loglik(double, itemmap&, double);
negative_binomial_params mle_negative_binomial(const triplets&,
                                               unsigned long int);
ztp_params mle_ztp(const triplets&, unsigned long int);
double calc_auc(const predictions&);
void calc_metrics(const vector<double>&, predictions&, vector<double>&,
                  vector<double>&, vector<double>&, vector<double>&,
                  vector<double>&, int, int);
void write_h5vector(H5::H5File&, string&, vector<double>&);
void write_h5vector(H5::H5File&, string&, vector<unsigned long int>&);
void write_h5vector(H5::H5File&, string&, vector<int>& );
void write_h5array2d(H5::H5File&, string&, array2d&);
void read_h5vector(H5::H5File&, string&, vector<double>&);
void read_h5vector(H5::H5File&, string&, vector<unsigned long int>&);
void read_h5vector(H5::H5File&, string&, vector<int>&);
void read_h5array2d(H5::H5File&, string&, array2d&);
void add_vector_to_ptree(bptree::ptree&, string&, vector<double>&, int);
#endif
