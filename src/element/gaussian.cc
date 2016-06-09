#include "utils.h"
#include "element/gaussian.h"
using namespace std;
namespace bmath = boost::math;

GaussianHyperParamStats::GaussianHyperParamStats() {}

void GaussianHyperParamStats::initialize(double mu_0, double sigma2_0) {
  // Normal-Gaussian prior
  mu_0 = mu_0;
  a_0 = 1.0;
  b_0 = sigma2_0;
  lambda_0 = 1.0;
  sum_n = 0;    // N
  sum_phi = 0;  // \sum E[phi1^2/phi2]
  sum_x = 0;    // \sum y E[phi1/phi2]
  sum_x2 = 0;   // \sum y^2 E[1/phi2]
}

void GaussianHyperParamStats::update_sum_n(double n = 1.0) { sum_n += n; }

void GaussianHyperParamStats::update_sum_phi(double e_phi1sq_div_phi2 = 1.0) {
  // variational update
  sum_phi += e_phi1sq_div_phi2;
}

void GaussianHyperParamStats::collapsed_update_sum_phi(double phi1 = 1.0,
                                                       double phi2 = 1.0) {
  // collapsed variational update
  sum_phi += pow(phi1, 2) / phi2;
}

void GaussianHyperParamStats::update_sum_x(double val,
                                           double e_phi1_div_phi2 = 1.0) {
  // variational update
  sum_x += val * e_phi1_div_phi2;
}

void GaussianHyperParamStats::collapsed_update_sum_x(double val,
                                                     double phi1 = 1.0,
                                                     double phi2 = 1.0) {
  // collapsed variational update
  sum_x += val * phi1 / phi2;
}

void GaussianHyperParamStats::update_sum_x2(double val,
                                            double e_1_div_phi2 = 1.0) {
  // variational update
  sum_x2 += pow(val, 2) * e_1_div_phi2;
}

void GaussianHyperParamStats::collapsed_update_sum_x2(double val,
                                                      double phi2 = 1.0) {
  // collapsed variational update
  sum_x2 += pow(val, 2) / phi2;
}

double GaussianHyperParamStats::get_mu_gradient() {
  return (lambda_0 * mu_0 + sum_x) / (lambda_0 + sum_phi);
}

double GaussianHyperParamStats::get_sigma2_gradient(double a, double b) {
  return b / a / (lambda_0 + sum_phi);
}

double GaussianHyperParamStats::get_a_gradient() {
  return a_0 + (sum_n + 1.0) / 2.0;
}

double GaussianHyperParamStats::get_b_gradient(double mu, double sigma2) {
  return b_0 +
         0.5 * (sum_x2 + lambda_0 * pow(mu_0, 2) -
                2 * mu * (lambda_0 * mu_0 + sum_x) +
                (pow(mu, 2) + sigma2) * (lambda_0 + sum_phi));
}

Gaussian::Gaussian() {}

Gaussian::Gaussian(double mu_, double sigma2_, int n_trunc_, double xi_,
                   double tau_) {
  this->initialize(mu_, sigma2_, n_trunc_, xi_, tau_);
}

void Gaussian::initialize(double mu_, double sigma2_, int n_trunc_, double xi_,
                          double tau_) {
  mu = mu_;
  mu_0 = mu;
  sigma2 = sigma2_;
  sigma2_0 = sigma2;
  rho_a = 1.0;
  rho_b = sigma2;
  mu_mu = mu;
  mu_sigma2 = rho_b / rho_a;
  n_trunc = n_trunc_;
  xi = xi_;
  tau = tau_;
  t_hyper = tau;
  cache_log_fact.resize(n_trunc);
  for (int j = 0; j < n_trunc; ++j) cache_log_fact[j] = bmath::lgamma(j + 1);
  set_mu(mu);
}

void Gaussian::initialize_with_sparse_matrix(SparseMatrix& sm, int n_trunc_,
                                             double xi_ = 0.7,
                                             double tau_ = 10000) {
  const triplets& ind2train = sm.get_ind2train();
  normal_params params = mle_normal(ind2train, 1000);
  this->initialize(params.mu, params.sigma2, n_trunc_, xi_, tau_);
}

void Gaussian::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                             int n_trunc_, double xi_ = 0.7,
                                             double tau_ = 10000) {
  initialize_with_sparse_matrix(sm, n_trunc_, xi_, tau_);
}

void Gaussian::serialize(string& fname) {
  H5::H5File file(fname, H5F_ACC_TRUNC);
  vector<unsigned long int> uiparams(1);
  uiparams[0] = t_hyper;
  string dsetname = "uiparams";
  write_h5vector(file, dsetname, uiparams);

  vector<int> iparams(1);
  iparams[0] = n_trunc;
  dsetname = "iparams";
  write_h5vector(file, dsetname, iparams);

  vector<double> params(10);
  params[0] = mu;
  params[1] = mu_0;
  params[2] = sigma2;
  params[3] = sigma2_0;
  params[4] = rho_a;
  params[5] = rho_b;
  params[6] = mu_mu;
  params[7] = mu_sigma2;
  params[8] = xi;
  params[9] = tau;
  dsetname = "params";
  write_h5vector(file, dsetname, params);
  file.close();
}

void Gaussian::deserialize(string& fname) {
  H5::H5File file(fname, H5F_ACC_RDONLY);
  vector<unsigned long int> uiparams(1);
  string dsetname = "uiparams";
  read_h5vector(file, dsetname, uiparams);
  unsigned long int t_hyper_ = uiparams[0];

  vector<int> iparams(1);
  dsetname = "iparams";
  read_h5vector(file, dsetname, iparams);
  int n_trunc_ = iparams[0];

  vector<double> params(10);
  dsetname = "params";
  read_h5vector(file, dsetname, params);
  double mu_ = params[0];
  double mu_0_ = params[1];
  double sigma2_ = params[2];
  double sigma2_0_ = params[3];
  double rho_a_ = params[4];
  double rho_b_ = params[5];
  double mu_mu_ = params[6];
  double mu_sigma2_ = params[7];
  double xi_ = params[7];
  double tau_ = params[7];
  this->initialize(mu_, sigma2_, n_trunc_, xi_, tau_);
  mu_0 = mu_0_;
  sigma2_0 = sigma2_0_;
  rho_a = rho_a_;
  rho_b = rho_b_;
  mu_mu = mu_mu_;
  mu_sigma2 = mu_sigma2_;
  t_hyper = t_hyper_;
  file.close();
}

double Gaussian::get_mu() { return mu; }

double Gaussian::get_sigma2() { return sigma2; }

int Gaussian::get_n_trunc() { return n_trunc; }

int Gaussian::get_n_component() { return 1; }

void Gaussian::set_mu(double mu_) {
  mu = mu_;
  element_mean = mu;
}

void Gaussian::set_sigma2(double a_, double b_) { sigma2 = b_ / a_; }

HyperParamStats* Gaussian::create_hyperparam_stats() {
  GaussianHyperParamStats* hps = new GaussianHyperParamStats();
  hps->initialize(mu_0, sigma2_0);
  return hps;
}

void Gaussian::update_hyperparam(HyperParamStats* stats) {
  update_hyperparam(dynamic_cast<GaussianHyperParamStats*>(stats));
}

void Gaussian::update_hyperparam(GaussianHyperParamStats* stats) {
  t_hyper += 1;
  double lr = pow(t_hyper, -xi);
  mu_mu = (1 - lr) * mu_mu + lr * stats->get_mu_gradient();
  mu_sigma2 =
      (1 - lr) * mu_sigma2 + lr * stats->get_sigma2_gradient(rho_a, rho_b);
  rho_a = (1 - lr) * rho_a + lr * stats->get_a_gradient();
  rho_b = (1 - lr) * rho_b + lr * stats->get_b_gradient(mu_mu, mu_sigma2);
  this->set_mu(mu_mu);
  this->set_sigma2(rho_a, rho_b);
  stats->initialize(mu_0, sigma2_0);
}

double Gaussian::predict() { return element_mean; }

double Gaussian::predict(unsigned long int ui, unsigned long int ii,
                         vector<double>& buffer_k) {
  return element_mean;
}

void Gaussian::update(double val, GaussianHyperParamStats* stats) {
  stats->update_sum_n(1.0);
  stats->collapsed_update_sum_phi(1.0, 1.0);
  stats->collapsed_update_sum_x(val, 1.0, 1.0);
  stats->collapsed_update_sum_x2(val, 1.0);
}

void Gaussian::update(unsigned long int ui, unsigned long int ii, double val,
                      vector<double>& buffer_k, vector<double>& buffer_n,
                      HyperParamStats* stats) {
  update(val, dynamic_cast<GaussianHyperParamStats*>(stats));
}

void Gaussian::test(double val, test_result& res) {
  double pred = element_mean;
  double sqerr = pow(val - pred, 2);
  double tll = -0.5 * sqerr / sigma2 -
               0.5 * log(2 * boost::math::constants::pi<double>() * sigma2);
  res.pred = pred;
  res.test_loglik = tll;
  res.test_error = sqerr;
  res.cond_pred = pred;
  res.cond_test_loglik = tll;
  res.cond_test_error = sqerr;
}

void Gaussian::test(unsigned long int ui, unsigned long int ii, double val,
                    vector<double>& buffer_k, vector<double>& buffer_n,
                    test_result& res) {
  test(val, res);
}

void Gaussian::update_q_n(double log_lambda, double val,
                          vector<double>& phi_mean, vector<double>& phi_var,
                          vector<double>& q_n) {
  for (int n = 1; n < n_trunc; ++n)
    q_n[n - 1] = -0.5 * pow(val - mu * phi_mean[n], 2) / sigma2 / phi_var[n] -
                 0.5 * log(2 * boost::math::constants::pi<double>() * sigma2 *
                           phi_var[n]) -
                 cache_log_fact[n] + n * log_lambda;
  double norm = logsumexp(q_n, n_trunc - 1);
  for (int n = n_trunc - 1; n != 0; --n) q_n[n] = exp(q_n[n - 1] - norm);
  q_n[0] = 0.0;
}

void Gaussian::update_q_n(unsigned long int ui, unsigned long int ii,
                          double log_lambda, double val,
                          vector<double>& phi_mean, vector<double>& phi_var,
                          vector<double>& buffer_k, vector<double>& q_n) {
  update_q_n(log_lambda, val, phi_mean, phi_var, q_n);
}

void Gaussian::update(double val, GaussianHyperParamStats* stats,
                      double e_phi_mean, double e_phi_var) {
  stats->update_sum_n(1);
  stats->collapsed_update_sum_phi(e_phi_mean, e_phi_var);
  stats->collapsed_update_sum_x(val, e_phi_mean, e_phi_var);
  stats->collapsed_update_sum_x2(val, e_phi_var);
}

void Gaussian::update(unsigned long int ui, unsigned long int ii, double val,
                      vector<double>& buffer_k, vector<double>& buffer_n,
                      HyperParamStats* stats, double e_phi_mean,
                      double e_phi_var) {
  update(val, dynamic_cast<GaussianHyperParamStats*>(stats), e_phi_mean,
         e_phi_var);
}

void Gaussian::update_test_result(double log_lambda, double val,
                                  vector<double>& buffer_n,
                                  vector<double>& phi_mean,
                                  vector<double>& phi_var, test_result& res) {
  double m = exp(log_lambda);
  double cond_norm = bmath::expm1(m);
  double pred = element_mean * m;
  double cond_pred = element_mean * m * exp(m) / cond_norm;
  double tll = 0.0;
  if (val != 0) {
    for (int n = 1; n < n_trunc; ++n)
      buffer_n[n - 1] =
          -0.5 * pow(val - mu * phi_mean[n], 2) / sigma2 / phi_var[n] -
          0.5 * log(2 * bmath::constants::pi<double>() * sigma2 * phi_var[n]) -
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

void Gaussian::update_test_result(unsigned long int ui, unsigned long int ii,
                                  double log_lambda, double val,
                                  vector<double>& buffer_k,
                                  vector<double>& buffer_n,
                                  vector<double>& phi_mean,
                                  vector<double>& phi_var, test_result& res) {
  update_test_result(log_lambda, val, buffer_n, phi_mean, phi_var, res);
}
