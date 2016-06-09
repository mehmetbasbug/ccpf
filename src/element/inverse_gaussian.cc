#include "utils.h"
#include "element/inverse_gaussian.h"
using namespace std;
namespace bmath = boost::math;

InverseGaussian::InverseGaussian() {}

InverseGaussian::InverseGaussian(double mu_, double lambda_, int n_trunc_,
                                 double xi_, double tau_) {
  this->initialize(mu_, lambda_, n_trunc_, xi_, tau_);
}

void InverseGaussian::initialize(double mu_, double lambda_, int n_trunc_,
                                 double xi_, double tau_) {
  mu = mu_;
  mu_0 = mu;
  lambda = lambda_;
  lambda_0 = lambda;
  n_trunc = n_trunc_;
  xi = xi_;
  tau = tau_;
  t_hyper = tau;
  cache_log_fact.resize(n_trunc);
  for (int j = 0; j < n_trunc; ++j) cache_log_fact[j] = bmath::lgamma(j + 1);
  this->set_mu(mu);
}

void InverseGaussian::initialize_with_sparse_matrix(SparseMatrix& sm,
                                                    int n_trunc_,
                                                    double xi_ = 0.7,
                                                    double tau_ = 10000) {
  const triplets& ind2train = sm.get_ind2train();
  inverse_gaussian_params params = mle_inverse_gaussian(ind2train, 1000);
  this->initialize(params.mu, params.lambda, n_trunc_, xi_, tau_);
}

void InverseGaussian::initialize_with_sparse_matrix(SparseMatrix& sm,
                                                    int n_component_,
                                                    int n_trunc_,
                                                    double xi_ = 0.7,
                                                    double tau_ = 10000) {
  initialize_with_sparse_matrix(sm, n_trunc_, xi_, tau_);
}

void InverseGaussian::serialize(string& fname) {
  H5::H5File file(fname, H5F_ACC_TRUNC);
  vector<unsigned long int> uiparams(1);
  uiparams[0] = t_hyper;
  string dsetname = "uiparams";
  write_h5vector(file, dsetname, uiparams);

  vector<int> iparams(1);
  iparams[0] = n_trunc;
  dsetname = "iparams";
  write_h5vector(file, dsetname, iparams);

  vector<double> params(6);
  params[0] = mu;
  params[1] = mu_0;
  params[2] = lambda;
  params[3] = lambda_0;
  params[4] = xi;
  params[5] = tau;
  dsetname = "params";
  write_h5vector(file, dsetname, params);
  file.close();
}

void InverseGaussian::deserialize(string& fname) {
  H5::H5File file(fname, H5F_ACC_RDONLY);
  vector<unsigned long int> uiparams(1);
  string dsetname = "uiparams";
  read_h5vector(file, dsetname, uiparams);
  unsigned long int t_hyper_ = uiparams[0];

  vector<int> iparams(1);
  dsetname = "iparams";
  read_h5vector(file, dsetname, iparams);
  int n_trunc_ = iparams[0];

  vector<double> params(6);
  dsetname = "params";
  read_h5vector(file, dsetname, params);
  double mu_ = params[0];
  double mu_0_ = params[1];
  double lambda_ = params[2];
  double lambda_0_ = params[3];
  double xi_ = params[4];
  double tau_ = params[5];
  this->initialize(mu_, lambda_, n_trunc_, xi_, tau_);
  mu_0 = mu_0_;
  lambda_0 = lambda_0_;
  t_hyper = t_hyper_;
  file.close();
}
double InverseGaussian::get_mu() { return mu; }

double InverseGaussian::get_lambda() { return lambda; }

int InverseGaussian::get_n_trunc() { return n_trunc; }

int InverseGaussian::get_n_component() { return 1; }

void InverseGaussian::set_mu(double mu_) {
  mu = mu_;
  element_mean = mu;
}

void InverseGaussian::set_lambda(double lambda_) { lambda = lambda_; }

HyperParamStats* InverseGaussian::create_hyperparam_stats() {
  NullHyperParamStats* hps = new NullHyperParamStats();
  return hps;
}

void InverseGaussian::update_hyperparam(HyperParamStats* stats) {}

double InverseGaussian::predict() { return element_mean; }

double InverseGaussian::predict(unsigned long int ui, unsigned long int ii,
                                vector<double>& buffer_k) {
  return predict();
}

void InverseGaussian::update(unsigned long int ui, unsigned long int ii,
                             double val, vector<double>& buffer_k,
                             vector<double>& buffer_n, HyperParamStats* stats) {
}

void InverseGaussian::test(double val, test_result& res) {
  double pred = element_mean;
  double sqerr = pow(val - pred, 2);
  double tll = -0.5 * lambda * sqerr / mu / mu / val + 0.5 * log(lambda) -
               0.5 * log(2 * boost::math::constants::pi<double>()) - 1.5 * log(val);
  res.pred = pred;
  res.test_loglik = tll;
  res.test_error = sqerr;
  res.cond_pred = pred;
  res.cond_test_loglik = tll;
  res.cond_test_error = sqerr;
}

void InverseGaussian::test(unsigned long int ui, unsigned long int ii,
                           double val, vector<double>& buffer_k,
                           vector<double>& buffer_n, test_result& res) {
  test(val, res);
}

void InverseGaussian::update_q_n(double log_lambda, double val,
                                 vector<double>& phi, vector<double>& q_n) {
  for (int n = 1; n < n_trunc; ++n)
    q_n[n - 1] = lambda * phi[n] / mu - 0.5 * phi[n] * phi[n] * lambda / val +
                 +0.5 * log(lambda) + log(phi[n]) + n * log_lambda -
                 cache_log_fact[n];
  double norm = logsumexp(q_n, n_trunc - 1);
  for (int n = n_trunc - 1; n !=0; --n) q_n[n] = exp(q_n[n-1] - norm);
  q_n[0] = 0.0;
}

void InverseGaussian::update_q_n(unsigned long int ui, unsigned long int ii,
                                 double log_lambda, double val,
                                 vector<double>& phi,
                                 vector<double>& buffer_k,
                                 vector<double>& q_n) {
  update_q_n(log_lambda, val, phi, q_n);
}

void InverseGaussian::update(unsigned long int ui, unsigned long int ii,
                             double val, vector<double>& buffer_k,
                             vector<double>& buffer_n, HyperParamStats* stats,
                             double e_phi) {}

void InverseGaussian::update_test_result(double log_lambda, double val,
                                         vector<double>& buffer_n,
                                         vector<double>& phi,
                                         test_result& res) {
  double m = exp(log_lambda);
  double cond_norm = bmath::expm1(m);
  double pred = element_mean * m;
  double cond_pred = element_mean * m * exp(m) / cond_norm;
  double tll = 0.0;
  if (val != 0) {
    for (int n = 1; n < n_trunc; ++n)
      buffer_n[n - 1] =
          -0.5 * lambda * pow(val - mu * phi[n], 2) / mu / mu / val +
          0.5 * log(lambda) + log(phi[n]) - 0.5 * log(2 * boost::math::constants::pi<double>()) -
          1.5 * log(val) + n * log_lambda - cache_log_fact[n];
    tll = logsumexp(buffer_n, n_trunc - 1);
  }
  res.pred = pred;
  res.test_loglik = tll - m;
  res.test_error = pow(val - pred, 2);
  res.cond_pred = cond_pred;
  res.cond_test_loglik = tll - log(cond_norm);
  res.cond_test_error = pow(val - cond_pred, 2);
}

void InverseGaussian::update_test_result(
    unsigned long int ui, unsigned long int ii, double log_lambda, double val,
    vector<double>& buffer_k, vector<double>& buffer_n,
    vector<double>& phi, test_result& res) {
  update_test_result(log_lambda, val, buffer_n, phi, res);
}
