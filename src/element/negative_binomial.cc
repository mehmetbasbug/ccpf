#include "utils.h"
#include "element/negative_binomial.h"
using namespace std;
namespace bmath = boost::math;

NegativeBinomialHyperParamStats::NegativeBinomialHyperParamStats() {}

void NegativeBinomialHyperParamStats::initialize(double p_0) {
  p_alpha = 1.0;
  p_beta = 1.0 / p_0 - 1;
}

void NegativeBinomialHyperParamStats::update_p_alpha(double sum_x) {
  p_alpha += sum_x;
}

void NegativeBinomialHyperParamStats::update_p_beta(double sum_r) {
  p_beta += sum_r;
}

double NegativeBinomialHyperParamStats::get_p_gradient() {
  return p_alpha / (p_alpha + p_beta);
}

NegativeBinomial::NegativeBinomial() {}

NegativeBinomial::NegativeBinomial(double p_, double r_, int n_trunc_,
                                   int val_max_, double xi_, double tau_) {
  this->initialize(p_, r_, n_trunc_, val_max_, xi_, tau_);
}

void NegativeBinomial::initialize(double p_, double r_, int n_trunc_,
                                  int val_max_, double xi_, double tau_) {
  p = p_;
  p_0 = p;
  r = r_;
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
  this->set_p(p);
}

void NegativeBinomial::initialize_with_sparse_matrix(SparseMatrix& sm,
                                                     int n_trunc_,
                                                     double xi_ = 0.7,
                                                     double tau_ = 10000) {
  const triplets& ind2train = sm.get_ind2train();
  negative_binomial_params params = mle_negative_binomial(ind2train, 1000);
  this->initialize(params.p, params.r, n_trunc_, sm.get_max_response() * 2, xi_,
                   tau_);
}

void NegativeBinomial::initialize_with_sparse_matrix(SparseMatrix& sm,
                                                     int n_component_,
                                                     int n_trunc_,
                                                     double xi_ = 0.7,
                                                     double tau_ = 10000) {
  initialize_with_sparse_matrix(sm, n_trunc_, xi_, tau_);
}

void NegativeBinomial::serialize(string& fname) {
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

  vector<double> params(5);
  params[0] = p;
  params[1] = p_0;
  params[2] = r;
  params[3] = xi;
  params[4] = tau;
  dsetname = "params";
  write_h5vector(file, dsetname, params);
  file.close();
}

void NegativeBinomial::deserialize(string& fname) {
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

  vector<double> params(5);
  dsetname = "params";
  read_h5vector(file, dsetname, params);
  double p_ = params[0];
  double p_0_ = params[1];
  double r_ = params[2];
  double xi_ = params[3];
  double tau_ = params[4];
  this->initialize(p_, r_, n_trunc_, val_max_, xi_, tau_);
  p_0 = p_0_;
  t_hyper = t_hyper_;
  file.close();
}

double NegativeBinomial::get_p() { return p; }

double NegativeBinomial::get_r() { return r; }

int NegativeBinomial::get_n_trunc() { return n_trunc; }

int NegativeBinomial::get_n_component() { return 1; }

void NegativeBinomial::set_p(double p_) {
  p = p_;
  element_mean = p * r / (1.0 - p);
  for (int val = 0; val < val_max; ++val)
    for (int n = 1; n < n_trunc; ++n)
      cache_qvar[val][n] = bmath::lgamma(n * r + val) - cache_log_fact[val] -
                           bmath::lgamma(n * r) + n * r * log(1 - p) +
                           val * log(p) - cache_log_fact[n];
}

HyperParamStats* NegativeBinomial::create_hyperparam_stats() {
  NegativeBinomialHyperParamStats* hps = new NegativeBinomialHyperParamStats();
  hps->initialize(p_0);
  return hps;
}

void NegativeBinomial::update_hyperparam(
    NegativeBinomialHyperParamStats* stats) {
  t_hyper += 1;
  double lr = pow(t_hyper, -xi);
  p = (1 - lr) * p + lr * stats->get_p_gradient();
  this->set_p(p);
  stats->initialize(p_0);
}

void NegativeBinomial::update_hyperparam(HyperParamStats* stats) {
  update_hyperparam(dynamic_cast<NegativeBinomialHyperParamStats*>(stats));
}

double NegativeBinomial::predict() { return element_mean; }

double NegativeBinomial::predict(unsigned long int ui, unsigned long int ii,
                                 vector<double>& buffer_k) {
  return predict();
}

void NegativeBinomial::update(int val, NegativeBinomialHyperParamStats* stats) {
  stats->update_p_alpha(val);
  stats->update_p_beta(r);
}

void NegativeBinomial::update(unsigned long int ui, unsigned long int ii,
                              double val, vector<double>& buffer_k,
                              vector<double>& buffer_n,
                              HyperParamStats* stats) {
  update(val, dynamic_cast<NegativeBinomialHyperParamStats*>(stats));
}

void NegativeBinomial::test(int val, test_result& res) {
  double pred = element_mean;
  double tll = bmath::lgamma(r + val) - cache_log_fact[val] - bmath::lgamma(r) +
               r * log(1 - p) + val * log(p);
  res.pred = pred;
  res.test_loglik = tll;
  res.test_error = pow(val - pred, 2);
  res.cond_pred = pred;
  res.cond_test_loglik = tll;
  res.cond_test_error = pow(val - pred, 2);
}

void NegativeBinomial::test(unsigned long int ui, unsigned long int ii,
                            double val, vector<double>& buffer_k,
                            vector<double>& buffer_n, test_result& res) {
  test(val, res);
}

void NegativeBinomial::update_q_n(double log_lambda, int val,
                                  vector<double>& phi, vector<double>& q_n) {
  for (int n = 1; n < n_trunc; ++n)
    q_n[n - 1] = bmath::lgamma(phi[n] * r + val) - cache_log_fact[val] -
                 bmath::lgamma(phi[n] * r) + phi[n] * r * log(1 - p) +
                 val * log(p) + n * log_lambda - cache_log_fact[n];
  double norm = logsumexp(q_n, n_trunc - 1);
  for (int n = n_trunc - 1; n != 0; --n) q_n[n] = exp(q_n[n - 1] - norm);
  q_n[0] = 0.0;
}

void NegativeBinomial::update_q_n(unsigned long int ui, unsigned long int ii,
                                  double log_lambda, double val,
                                  vector<double>& phi, vector<double>& buffer_k,
                                  vector<double>& q_n) {
  update_q_n(log_lambda, (int)val, phi, q_n);
}

void NegativeBinomial::update(int val, NegativeBinomialHyperParamStats* stats,
                              double e_phi) {
  stats->update_p_alpha(val);
  stats->update_p_beta(e_phi * r);
}

void NegativeBinomial::update(unsigned long int ui, unsigned long int ii,
                              double val, vector<double>& buffer_k,
                              vector<double>& buffer_n, HyperParamStats* stats,
                              double e_phi) {
  update(val, dynamic_cast<NegativeBinomialHyperParamStats*>(stats), e_phi);
}

void NegativeBinomial::update_test_result(double log_lambda, int val,
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
      buffer_n[n - 1] = bmath::lgamma(phi[n] * r + val) - cache_log_fact[val] -
                        bmath::lgamma(phi[n] * r) + phi[n] * r * log(1 - p) +
                        val * log(p) + n * log_lambda - cache_log_fact[n];
    tll = logsumexp(buffer_n, n_trunc - 1);
  }
  res.pred = pred;
  res.test_loglik = tll - m;
  res.test_error = pow(val - pred, 2);
  res.cond_pred = cond_pred;
  res.cond_test_loglik = tll - log(cond_norm);
  res.cond_test_error = pow(val - cond_pred, 2);
}

void NegativeBinomial::update_test_result(
    unsigned long int ui, unsigned long int ii, double log_lambda, double val,
    vector<double>& buffer_k, vector<double>& buffer_n, vector<double>& phi,
    test_result& res) {
  update_test_result(log_lambda, (int)val, buffer_n, phi, res);
}
