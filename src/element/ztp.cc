#include "utils.h"
#include "element/ztp.h"
using namespace std;
namespace bmath = boost::math;

ZTPHyperParamStats::ZTPHyperParamStats() {}
void ZTPHyperParamStats::initialize(double lambda_0) {
  // Gamma priors on ZTP
  lambda_a = lambda_0;
  lambda_b = 1;
}
void ZTPHyperParamStats::update_lambda_a(double sum_x) { lambda_a += sum_x; }
void ZTPHyperParamStats::update_lambda_b(double sum_n) { lambda_b += sum_n; }
double ZTPHyperParamStats::get_lambda_gradient() {
  return lambda_a / lambda_b;
};

ZTP::ZTP() {}
ZTP::ZTP(double lambda_, int n_trunc_, int val_max_, double xi_, double tau_) {
  this->initialize(lambda_, n_trunc_, val_max_, xi_, tau_);
}

void ZTP::initialize(double lambda_, int n_trunc_, int val_max_, double xi_,
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

void ZTP::initialize_with_sparse_matrix(SparseMatrix& sm, int n_trunc_,
                                        double xi_ = 0.7, double tau_ = 10000) {
  const triplets& ind2train = sm.get_ind2train();
  ztp_params params = mle_ztp(ind2train, 1000);
  this->initialize(params.lambda, n_trunc_, sm.get_max_response() * 2, xi_,
                   tau_);
}

void ZTP::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                        int n_trunc_, double xi_ = 0.7,
                                        double tau_ = 10000) {
  initialize_with_sparse_matrix(sm, n_trunc_, xi_, tau_);
}

void ZTP::serialize(string& fname) {
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

void ZTP::deserialize(string& fname) {
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

double ZTP::get_lambda() { return lambda; }

int ZTP::get_n_trunc() { return n_trunc; }

int ZTP::get_n_component() { return 1; }

void ZTP::set_lambda(double lambda_) {
  lambda = lambda_;
  element_mean = lambda * exp(lambda) / bmath::expm1(lambda);
  double log_lambda = log(lambda);
  double expm1lambda = bmath::expm1(lambda);
  vector<double> qntmp(n_trunc, 0);
  for (int val = 1; val < val_max; ++val) {
    double tmp = val * log_lambda - bmath::lgamma(val + 1);
    for (int n = 1; n < n_trunc; ++n) {
      for (int j = 0; j < n; ++j)
        qntmp[j] = val * log(n - j) - cache_log_fact[j] - cache_log_fact[n - j];
      cache_qvar[val][n] = logsumexp2(qntmp, n) - n * log(expm1lambda) + tmp;
    }
  }
}

HyperParamStats* ZTP::create_hyperparam_stats() {
  ZTPHyperParamStats* hps = new ZTPHyperParamStats();
  hps->initialize(lambda_0);
  return hps;
}

void ZTP::update_hyperparam(HyperParamStats* stats) {
  update_hyperparam(dynamic_cast<ZTPHyperParamStats*>(stats));
}

void ZTP::update_hyperparam(ZTPHyperParamStats* stats) {
  t_hyper += 1;
  double lr = pow(t_hyper, -xi);
  lambda = (1 - lr) * lambda + lr * stats->get_lambda_gradient();
  this->set_lambda(lambda);
  stats->initialize(lambda_0);
}

double ZTP::predict() { return element_mean; }

double ZTP::predict(unsigned long int ui, unsigned long int ii,
                    vector<double>& buffer_k) {
  return predict();
}

void ZTP::update(int val, ZTPHyperParamStats* stats) {
  stats->update_lambda_a(val);
  stats->update_lambda_b(1);
}

void ZTP::update(unsigned long int ui, unsigned long int ii, double val,
                 vector<double>& buffer_k, vector<double>& buffer_n,
                 HyperParamStats* stats) {
  update((int)val, dynamic_cast<ZTPHyperParamStats*>(stats));
}

void ZTP::test(int val, test_result& res) {
  double pred = element_mean;
  double tll =
      val * log(lambda) - log(bmath::expm1(lambda)) - cache_log_fact[val];
  res.pred = pred;
  res.test_loglik = tll;
  res.test_error = pow(val - pred, 2);
  res.cond_pred = pred;
  res.cond_test_loglik = tll;
  res.cond_test_error = pow(val - pred, 2);
}

void ZTP::test(unsigned long int ui, unsigned long int ii, double val,
               vector<double>& buffer_k, vector<double>& buffer_n,
               test_result& res) {
  test((int)val, res);
}

void ZTP::update_q_n(double log_lambda, int val, vector<double>& q_n) {
  // For fixed phi(n) = n
  for (int n = 1; n < n_trunc; ++n)
    q_n[n - 1] = cache_qvar[val][n] + n * log_lambda;
  double norm = logsumexp(q_n, n_trunc - 1);
  for (int n = n_trunc - 1; n !=0; --n) q_n[n] = exp(q_n[n-1] - norm);
  q_n[0] = 0.0;
}

void ZTP::update_q_n(unsigned long int ui, unsigned long int ii,
                     double log_lambda, double val, vector<double>& phi,
                     vector<double>& buffer_k, vector<double>& q_n) {
  update_q_n(log_lambda, (int)val, q_n);
}

void ZTP::update(int val, ZTPHyperParamStats* stats, double e_phi) {
  stats->update_lambda_a(val);
  stats->update_lambda_b(e_phi);
}

void ZTP::update(unsigned long int ui, unsigned long int ii, double val,
                 vector<double>& buffer_k, vector<double>& buffer_n,
                 HyperParamStats* stats, double e_phi) {
  update((int)val, dynamic_cast<ZTPHyperParamStats*>(stats), e_phi);
}

void ZTP::update_test_result(double log_lambda, int val,
                             vector<double>& buffer_n, test_result& res) {
  double m = exp(log_lambda);
  double cond_norm = bmath::expm1(m);
  double pred = element_mean * m;
  double cond_pred = element_mean * m * exp(m) / cond_norm;
  double tll = 0.0;
  if (val != 0) {
    for (int n = 1; n < n_trunc; ++n)
      buffer_n[n - 1] = cache_qvar[val][n] + n * log_lambda;
    tll = logsumexp(buffer_n, n_trunc - 1);
  }
  res.pred = pred;
  res.test_loglik = tll - m;
  res.test_error = pow(val - pred, 2);
  res.cond_pred = cond_pred;
  res.cond_test_loglik = tll - log(cond_norm);
  res.cond_test_error = pow(val - cond_pred, 2);
}

void ZTP::update_test_result(unsigned long int ui, unsigned long int ii,
                             double log_lambda, double val,
                             vector<double>& buffer_k, vector<double>& buffer_n,
                             vector<double>& phi, test_result& res) {
  update_test_result(log_lambda, (int)val, buffer_n, res);
}
