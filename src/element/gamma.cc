#include "utils.h"
#include "element/gamma.h"
using namespace std;
namespace bmath = boost::math;

GammaHyperParamStats::GammaHyperParamStats() {}

void GammaHyperParamStats::initialize(double beta_0) {
  beta_a = beta_0;
  beta_b = 1.0;
}

void GammaHyperParamStats::update_beta_a(double sum_alpha) {
  beta_a += sum_alpha;
}

void GammaHyperParamStats::update_beta_b(double sum_x) { beta_b += sum_x; }

double GammaHyperParamStats::get_beta_gradient() { return beta_a / beta_b; }

Gamma::Gamma() {}

Gamma::Gamma(double alpha_, double beta_, int n_trunc_, double xi_,
             double tau_) {
  this->initialize(alpha_, beta_, n_trunc_, xi_, tau_);
}

void Gamma::initialize(double alpha_, double beta_, int n_trunc_, double xi_,
                       double tau_) {
  alpha = alpha_;
  alpha_0 = alpha;
  beta = beta_;
  beta_0 = beta;
  n_trunc = n_trunc_;
  xi = xi_;
  tau = tau_;
  t_hyper = tau;
  cache_log_fact.resize(n_trunc);
  cache_log_scaled_fact.resize(n_trunc);
  for (int j = 0; j < n_trunc; ++j) cache_log_fact[j] = bmath::lgamma(j + 1);
  this->set_alpha(alpha);
}

void Gamma::initialize_with_sparse_matrix(SparseMatrix& sm, int n_trunc_,
                                          double xi_ = 0.7,
                                          double tau_ = 10000) {
  const triplets& ind2train = sm.get_ind2train();
  gamma_params params = mle_gamma(ind2train, 1000);
  this->initialize(params.shape, params.rate, n_trunc_, xi_, tau_);
}

void Gamma::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                          int n_trunc_, double xi_ = 0.7,
                                          double tau_ = 10000) {
  initialize_with_sparse_matrix(sm, n_trunc_, xi_, tau_);
}

void Gamma::serialize(string& fname) {
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
  params[0] = alpha;
  params[1] = alpha_0;
  params[2] = beta;
  params[3] = beta_0;
  params[4] = xi;
  params[5] = tau;
  dsetname = "params";
  write_h5vector(file, dsetname, params);
  file.close();
}

void Gamma::deserialize(string& fname) {
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
  double alpha_ = params[0];
  double alpha_0_ = params[1];
  double beta_ = params[2];
  double beta_0_ = params[3];
  double xi_ = params[4];
  double tau_ = params[5];
  this->initialize(alpha_, beta_, n_trunc_, xi_, tau_);
  alpha_0 = alpha_0_;
  beta_0 = beta_0_;
  t_hyper = t_hyper_;
  file.close();
}

double Gamma::get_alpha() { return alpha; }

double Gamma::get_beta() { return beta; }

int Gamma::get_n_trunc() { return n_trunc; }

int Gamma::get_n_component() { return 1; }

void Gamma::set_alpha(double alpha_) {
  alpha = alpha_;
  element_mean = alpha / beta;
  for (int n = 1; n < n_trunc; ++n)
    cache_log_scaled_fact[n] = bmath::lgamma(n * alpha);
}

void Gamma::set_beta(double beta_) {
  beta = beta_;
  element_mean = alpha / beta;
}

HyperParamStats* Gamma::create_hyperparam_stats() {
  GammaHyperParamStats* hps = new GammaHyperParamStats();
  hps->initialize(beta_0);
  return hps;
}

void Gamma::update_hyperparam(HyperParamStats* stats) {
  update_hyperparam(dynamic_cast<GammaHyperParamStats*>(stats));
}

void Gamma::update_hyperparam(GammaHyperParamStats* stats) {
  t_hyper += 1;
  double lr = pow(t_hyper, -xi);
  beta = (1 - lr) * beta + lr * stats->get_beta_gradient();
  this->set_beta(beta);
  stats->initialize(beta_0);
}

double Gamma::predict() { return element_mean; }

double Gamma::predict(unsigned long int ui, unsigned long int ii,
                      vector<double>& buffer_k) {
  return predict();
}

void Gamma::update(double val, GammaHyperParamStats* stats) {
  stats->update_beta_a(alpha);
  stats->update_beta_b(val);
}

void Gamma::update(unsigned long int ui, unsigned long int ii, double val,
                   vector<double>& buffer_k, vector<double>& buffer_n,
                   HyperParamStats* stats) {
  update(val, dynamic_cast<GammaHyperParamStats*>(stats));
}

void Gamma::test(double val, test_result& res) {
  double pred = element_mean;
  double tll = alpha * log(beta) + (alpha - 1) * log(val) - beta * val -
               bmath::lgamma(alpha);
  res.pred = pred;
  res.test_loglik = tll;
  res.test_error = pow(val - pred, 2);
  res.cond_pred = pred;
  res.cond_test_loglik = tll;
  res.cond_test_error = pow(val - pred, 2);
}

void Gamma::test(unsigned long int ui, unsigned long int ii, double val,
                 vector<double>& buffer_k, vector<double>& buffer_n,
                 test_result& res) {
  test(val, res);
}

void Gamma::update_q_n(double log_lambda, double val, vector<double>& phi,
                       vector<double>& q_n) {
  for (int n = 1; n < n_trunc; ++n)
    q_n[n - 1] = phi[n] * alpha * log(beta) + (phi[n] * alpha - 1) * log(val) -
                 bmath::lgamma(phi[n] * alpha) - beta * val -
                 cache_log_fact[n] + n * log_lambda;
  double norm = logsumexp(q_n, n_trunc - 1);
  for (int n = n_trunc - 1; n !=0; --n) q_n[n] = exp(q_n[n-1] - norm);
  q_n[0] = 0.0;
}

void Gamma::update_q_n(unsigned long int ui, unsigned long int ii,
                       double log_lambda, double val, vector<double>& phi,
                       vector<double>& buffer_k, vector<double>& q_n) {
  update_q_n(log_lambda, val, phi, q_n);
}

void Gamma::update(double val, GammaHyperParamStats* stats, double e_phi) {
  stats->update_beta_a(e_phi * alpha);
  stats->update_beta_b(val);
}

void Gamma::update(unsigned long int ui, unsigned long int ii, double val,
                   vector<double>& buffer_k, vector<double>& buffer_n,
                   HyperParamStats* stats, double e_phi) {
  update(val, dynamic_cast<GammaHyperParamStats*>(stats), e_phi);
}

void Gamma::update_test_result(double log_lambda, double val,
                               vector<double>& buffer_n, vector<double>& phi,
                               test_result& res) {
  double m = exp(log_lambda);
  double cond_norm = bmath::expm1(m);
  double pred = element_mean * m;
  double cond_pred = element_mean * m * exp(m) / cond_norm;
  double tll = 0.0;
  if (val != 0) {
    for (int n = 1; n < n_trunc; ++n)
      buffer_n[n - 1] = phi[n] * alpha * log(beta) +
                        (phi[n] * alpha - 1) * log(val) - beta * val -
                        bmath::lgamma(phi[n] * alpha) - cache_log_fact[n] +
                        n * log_lambda;
    tll = logsumexp(buffer_n, n_trunc - 1);
  }
  res.pred = pred;
  res.test_loglik = tll - m;
  res.test_error = pow(val - pred, 2);
  res.cond_pred = cond_pred;
  res.cond_test_loglik = tll - log(cond_norm);
  res.cond_test_error = pow(val - cond_pred, 2);
}

void Gamma::update_test_result(unsigned long int ui, unsigned long int ii,
                               double log_lambda, double val,
                               vector<double>& buffer_k,
                               vector<double>& buffer_n, vector<double>& phi,
                               test_result& res) {
  update_test_result(log_lambda, val, buffer_n, phi, res);
}
