#include "utils.h"
#include "element/poisson.h"
using namespace std;
namespace bmath = boost::math;

PoissonHyperParamStats::PoissonHyperParamStats() {}
void PoissonHyperParamStats::initialize(double lambda_0) {
  // Gamma priors on Poisson
  lambda_a = lambda_0;
  lambda_b = 1;
}
void PoissonHyperParamStats::update_lambda_a(double sum_x) {
  lambda_a += sum_x;
}
void PoissonHyperParamStats::update_lambda_b(double sum_n) {
  lambda_b += sum_n;
}
double PoissonHyperParamStats::get_lambda_gradient() {
  return lambda_a / lambda_b;
};

Poisson::Poisson() {}
Poisson::Poisson(double lambda_, int n_trunc_, int val_max_, double xi_,
                 double tau_) {
  this->initialize(lambda_, n_trunc_, val_max_, xi_, tau_);
}

void Poisson::initialize(double lambda_, int n_trunc_, int val_max_, double xi_,
                         double tau_) {
  lambda = lambda_;
  lambda_0 = lambda;
  n_trunc = n_trunc_;
  val_max = val_max_;
  xi = xi_;
  tau = tau_;
  t_hyper = tau;
  int max_dim = max(val_max, n_trunc);
  cache_log_fact.resize(max_dim);
  cache_qvar.resize(val_max);
  for (int i = 0; i < val_max; ++i) cache_qvar[i].resize(n_trunc);
  for (int j = 0; j < max_dim; ++j) cache_log_fact[j] = bmath::lgamma(j + 1);
  this->set_lambda(lambda);
}

void Poisson::initialize_with_sparse_matrix(SparseMatrix& sm, int n_trunc_,
                                            double xi_ = 0.7,
                                            double tau_ = 10000) {
  const triplets& ind2train = sm.get_ind2train();
  poisson_params params = mle_poisson(ind2train, 1000);
  this->initialize(params.lambda, n_trunc_, sm.get_max_response() * 2, xi_,
                   tau_);
}

void Poisson::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                            int n_trunc_, double xi_ = 0.7,
                                            double tau_ = 10000) {
  initialize_with_sparse_matrix(sm, n_trunc_, xi_, tau_);
}

void Poisson::serialize(string& fname) {
  H5::H5File file(fname, H5F_ACC_TRUNC);
  vector<unsigned long int> uiparams(1);
  uiparams[0] = t_hyper;
  string dsetname = "uiparams";
  write_h5vector(file, dsetname, uiparams);

  vector<int> iparams(2);
  iparams[0] = n_trunc;
  iparams[1] = val_max;
  dsetname = "iparams";
  write_h5vector(file, dsetname, iparams);

  vector<double> params(4);
  params[0] = lambda;
  params[1] = lambda_0;
  params[2] = xi;
  params[3] = tau;
  dsetname = "params";
  write_h5vector(file, dsetname, params);
  file.close();
}

void Poisson::deserialize(string& fname) {
  H5::H5File file(fname, H5F_ACC_RDONLY);
  vector<unsigned long int> uiparams(1);
  string dsetname = "uiparams";
  read_h5vector(file, dsetname, uiparams);
  unsigned long int t_hyper_ = uiparams[0];

  vector<int> iparams(2);
  dsetname = "iparams";
  read_h5vector(file, dsetname, iparams);
  int n_trunc_ = iparams[0];
  int val_max_ = iparams[1];

  vector<double> params(4);
  dsetname = "params";
  read_h5vector(file, dsetname, params);
  double lambda_ = params[0];
  double lambda_0_ = params[1];
  double xi_ = params[2];
  double tau_ = params[3];
  this->initialize(lambda_, n_trunc_, val_max_, xi_, tau_);
  lambda_0 = lambda_0_;
  t_hyper = t_hyper_;
  file.close();
}

double Poisson::get_lambda() { return lambda; }

int Poisson::get_n_trunc() { return n_trunc; }

int Poisson::get_n_component() { return 1; }

void Poisson::set_lambda(double lambda_) {
  lambda = lambda_;
  element_mean = lambda;
  double log_lambda = log(lambda);
  for (int n = 1; n < n_trunc; ++n)
    for (int val = 0; val < val_max; ++val)
      cache_qvar[val][n] = val * log_lambda + val * log(n) - n * lambda -
                           cache_log_fact[val] - cache_log_fact[n];
}

HyperParamStats* Poisson::create_hyperparam_stats() {
  PoissonHyperParamStats* hps = new PoissonHyperParamStats();
  hps->initialize(lambda_0);
  return hps;
}

void Poisson::update_hyperparam(PoissonHyperParamStats* stats) {
  t_hyper += 1;
  double lr = pow(t_hyper, -xi);
  lambda = (1 - lr) * lambda + lr * stats->get_lambda_gradient();
  this->set_lambda(lambda);
  stats->initialize(lambda_0);
}

void Poisson::update_hyperparam(HyperParamStats* stats) {
  update_hyperparam(dynamic_cast<PoissonHyperParamStats*>(stats));
}

double Poisson::predict() { return element_mean; }

double Poisson::predict(unsigned long int ui, unsigned long int ii,
                        vector<double>& buffer_k) {
  return predict();
}

void Poisson::update(int val, PoissonHyperParamStats* stats) {
  stats->update_lambda_a(val);
  stats->update_lambda_b(1);
}

void Poisson::update(unsigned long int ui, unsigned long int ii, double val,
                     vector<double>& buffer_k, vector<double>& buffer_n,
                     HyperParamStats* stats) {
  update((int)val, dynamic_cast<PoissonHyperParamStats*>(stats));
}

void Poisson::test(int val, test_result& res) {
  double pred = element_mean;
  double tll = val * log(lambda) - lambda - cache_log_fact[val];
  res.pred = pred;
  res.test_loglik = tll;
  res.test_error = pow(val - pred, 2);
  res.cond_pred = pred;
  res.cond_test_loglik = tll;
  res.cond_test_error = pow(val - pred, 2);
}

void Poisson::test(unsigned long int ui, unsigned long int ii, double val,
                   vector<double>& buffer_k, vector<double>& buffer_n,
                   test_result& res) {
  test((int)val, res);
}

void Poisson::update_q_n(double log_lambda, int val, vector<double>& phi,
                         vector<double>& q_n) {
  for (int n = 1; n < n_trunc; ++n)
    q_n[n - 1] = val * log(lambda) + val * log(phi[n]) - lambda * phi[n] -
                 cache_log_fact[val] - cache_log_fact[n] + n * log_lambda;
  double norm = logsumexp(q_n, n_trunc - 1);
  for (int n = n_trunc - 1; n !=0; --n) q_n[n] = exp(q_n[n-1] - norm);
  q_n[0] = 0.0;
}

void Poisson::update_q_n(unsigned long int ui, unsigned long int ii,
                         double log_lambda, double val, vector<double>& phi,
                         vector<double>& buffer_k, vector<double>& q_n) {
  update_q_n(log_lambda, (int)val, phi, q_n);
}

void Poisson::update(int val, PoissonHyperParamStats* stats, double e_phi) {
  stats->update_lambda_a(val);
  stats->update_lambda_b(e_phi);
}

void Poisson::update(unsigned long int ui, unsigned long int ii, double val,
                     vector<double>& buffer_k, vector<double>& buffer_n,
                     HyperParamStats* stats, double e_phi) {
  update((int)val, dynamic_cast<PoissonHyperParamStats*>(stats), e_phi);
}

void Poisson::update_test_result(double log_lambda, int val,
                                 vector<double>& buffer_n, vector<double>& phi,
                                 test_result& res) {
  double m = exp(log_lambda);
  double cond_norm = bmath::expm1(m);
  double pred = element_mean * m;
  double cond_pred = element_mean * m * exp(m) / cond_norm;
  double tll = 0.0;
  if (val != 0) {
    for (int n = 1; n < n_trunc; ++n)
      buffer_n[n - 1] = val * log(lambda) + val * log(phi[n]) -
                        lambda * phi[n] - cache_log_fact[val] -
                        cache_log_fact[n] + n * log_lambda;
    tll = logsumexp(buffer_n, n_trunc - 1);
  }
  res.pred = pred;
  res.test_loglik = tll - m;
  res.test_error = pow(val - pred, 2);
  res.cond_pred = cond_pred;
  res.cond_test_loglik = tll - log(cond_norm);
  res.cond_test_error = pow(val - cond_pred, 2);
}

void Poisson::update_test_result(unsigned long int ui, unsigned long int ii,
                                 double log_lambda, double val,
                                 vector<double>& buffer_k,
                                 vector<double>& buffer_n, vector<double>& phi,
                                 test_result& res) {
  update_test_result(log_lambda, (int)val, buffer_n, phi, res);
}
