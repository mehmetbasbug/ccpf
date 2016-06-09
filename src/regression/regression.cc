#include "utils.h"
#include "regression/regression.h"
using namespace std;
namespace bmath = boost::math;

RegressionHyperParamStats::RegressionHyperParamStats() {}
void RegressionHyperParamStats::initialize(double sigma2_0, double rho_0) {
  // Inverse Gamma prior on variance
  sigma2_alpha = 2;
  sigma2_beta = sigma2_0;
  rho_alpha = 2;
  rho_beta = rho_0;
}
void RegressionHyperParamStats::update_sigma2_alpha(double x) {
  sigma2_alpha += x / 2;
}
void RegressionHyperParamStats::update_sigma2_beta(double x) {
  sigma2_beta += x / 2;
}
void RegressionHyperParamStats::update_rho_alpha(double x) {
  rho_alpha += x / 2;
}
void RegressionHyperParamStats::update_rho_beta(double x) { rho_beta += x / 2; }
double RegressionHyperParamStats::get_sigma2_gradient() {
  return sigma2_beta / (sigma2_alpha - 1);
}
double RegressionHyperParamStats::get_rho_gradient() {
  return rho_beta / (rho_alpha - 1);
}

Regression::Regression() {}
Regression::Regression(unsigned long int n_user_, unsigned long int n_item_,
                       int n_component_, double sparsity_, double rho_,
                       double sigma2_, double xi_, double tau_, int n_trunc_) {
  this->initialize(n_user_, n_item_, n_component_, sparsity_, rho_, sigma2_,
                   xi_, tau_, n_trunc_);
}

void Regression::initialize(unsigned long int n_user_,
                            unsigned long int n_item_, int n_component_,
                            double sparsity_, double rho_, double sigma2_,
                            double xi_, double tau_, int n_trunc_) {
  n_user = n_user_;
  sparsity = sparsity_;
  n_eff_user = (1 - sparsity) * n_user;
  n_item = n_item_;
  n_eff_item = (1 - sparsity) * n_item;
  n_component = n_component_;
  rho = rho_;
  rho_0 = rho;
  sigma2 = sigma2_;
  sigma2_0 = sigma2;
  n_trunc = n_trunc_;
  xi = xi_;
  tau = tau_;
  t_hyper = tau;
  t_learning.resize(n_user_);
  a_s.resize(boost::extents[n_user][n_component]);
  b_s.resize(boost::extents[n_user][n_component]);
  xreg.resize(boost::extents[n_item][n_component]);
  eta.resize(n_component);
  for (size_t i = 0; i < n_user; ++i) {
    t_learning[i] = tau;
    for (int j = 0; j < n_component; ++j) {
      a_s[i][j] = 0;
      b_s[i][j] = rho;
    }
  }
  cache_log_fact.resize(n_trunc);
  for (int j = 0; j < n_trunc; ++j) cache_log_fact[j] = bmath::lgamma(j + 1);
}

void Regression::initialize_with_sparse_matrix(SparseMatrix& sm,
                                               int n_component_, int n_trunc_,
                                               double xi_ = 0.7,
                                               double tau_ = 10000) {
  const triplets& ind2train = sm.get_ind2train();
  normal_params params = mle_normal(ind2train, 1000);
  double mu_ = params.mu;
  double sigma2_ = params.sigma2;
  double mu_k = mu_ / (double)n_component_;
  double sigma2_k = sigma2_ / (double)n_component_;
  double rho_ = abs(sqrt(sigma2_k + mu_k * mu_k) - mu_k);
  this->initialize(sm.get_n_user(), sm.get_n_item(), n_component_,
                   sm.get_sparsity(), rho_, sigma2_, xi_, tau_, n_trunc_);
}

void Regression::initialize_with_sparse_matrix(SparseMatrix& sm,
                                               int n_component_,
                                               double xi_ = 0.7,
                                               double tau_ = 10000) {
  initialize_with_sparse_matrix(sm, n_component_, 10, xi_, tau_);
}

void Regression::serialize(string& fname) {
  H5::H5File file(fname, H5F_ACC_TRUNC);
  vector<unsigned long int> uiparams(3);
  uiparams[0] = n_user;
  uiparams[1] = n_item;
  uiparams[2] = t_hyper;
  string dsetname = "uiparams";
  write_h5vector(file, dsetname, uiparams);

  vector<int> iparams(2);
  iparams[0] = n_component;
  iparams[1] = n_trunc;
  dsetname = "iparams";
  write_h5vector(file, dsetname, iparams);

  vector<double> params(7);
  params[0] = sparsity;
  params[1] = sigma2;
  params[2] = sigma2_0;
  params[3] = rho;
  params[4] = rho_0;
  params[5] = xi;
  params[6] = tau;
  dsetname = "params";
  write_h5vector(file, dsetname, params);
  dsetname = "a_s";
  write_h5array2d(file, dsetname, a_s);
  dsetname = "b_s";
  write_h5array2d(file, dsetname, b_s);
  dsetname = "t_learning";
  write_h5vector(file, dsetname, t_learning);
  file.close();
}

void Regression::deserialize(string& fname) {
  H5::H5File file(fname, H5F_ACC_RDONLY);
  vector<unsigned long int> uiparams(3);
  string dsetname = "uiparams";
  read_h5vector(file, dsetname, uiparams);
  unsigned long int n_user_ = uiparams[0];
  unsigned long int n_item_ = uiparams[1];
  unsigned long int t_hyper_ = uiparams[1];

  vector<int> iparams(2);
  dsetname = "iparams";
  read_h5vector(file, dsetname, iparams);
  int n_component_ = iparams[0];
  int n_trunc_ = iparams[1];

  vector<double> params(7);
  dsetname = "params";
  read_h5vector(file, dsetname, params);
  double sparsity_ = params[0];
  double sigma2_ = params[1];
  double sigma2_0_ = params[2];
  double rho_ = params[3];
  double rho_0_ = params[4];
  double xi_ = params[5];
  double tau_ = params[6];
  this->initialize(n_user_, n_item_, n_component_, sparsity_, rho_, sigma2_,
                   xi_, tau_, n_trunc_);
  dsetname = "a_s";
  read_h5array2d(file, dsetname, a_s);
  dsetname = "b_s";
  read_h5array2d(file, dsetname, b_s);
  dsetname = "t_learning";
  read_h5vector(file, dsetname, t_learning);
  sigma2_0 = sigma2_0_;
  rho_0 = rho_0_;
  t_hyper = t_hyper_;
  file.close();
}

double Regression::get_rho() { return rho; }

double Regression::get_rho_0() { return rho_0; }

double Regression::get_sigma2() { return sigma2; }

double Regression::get_sigma2_0() { return sigma2_0; }

int Regression::get_n_trunc() { return n_trunc; }

int Regression::get_n_component() { return n_component; }

int Regression::get_n_xreg() { return n_component; }

void Regression::set_n_xreg(int n_xreg_) {
  n_component = n_xreg_;
  a_s.resize(boost::extents[n_user][n_component]);
  b_s.resize(boost::extents[n_user][n_component]);
  xreg.resize(boost::extents[n_item][n_component]);
  eta.resize(n_component);
  for (size_t i = 0; i < n_user; ++i) {
    t_learning[i] = tau;
    for (int j = 0; j < n_component; ++j) {
      a_s[i][j] = 0;
      b_s[i][j] = rho;
    }
  }
}

void Regression::set_rho(double rho_) { rho = rho_; }

void Regression::set_rho_0(double rho_0_) { rho_0 = rho_0_; }

void Regression::set_sigma2(double sigma2_) { sigma2 = sigma2_; }

void Regression::set_sigma2_0(double sigma2_0_) { sigma2_0 = sigma2_0_; }

double Regression::calc_lru(unsigned long int ui) {
  t_learning[ui] += 1;
  return pow(t_learning[ui], -xi);
}

void Regression::update_a_s(unsigned long int ui, unsigned long int ii,
                            double lru, vector<double>& err,
                            double phi_mean = 1.0, double phi_var = 1.0) {
  double gr = 0.0;
  for (int j = 0; j < n_component; ++j) {
    gr = (eta[j] * phi_var * sigma2 +
          rho * phi_mean * n_eff_item * err[j] * xreg[ii][j]) /
         (sigma2 * phi_var + rho * phi_mean * n_eff_item * pow(xreg[ii][j], 2));
    a_s[ui][j] = (1 - lru) * a_s[ui][j] + lru * gr;
  }
}

void Regression::update_b_s(unsigned long int ui, unsigned long int ii,
                            double lru, double phi_mean = 1.0,
                            double phi_var = 1.0) {
  double gr = 0.0;
  for (int j = 0; j < n_component; ++j) {
    gr = sigma2 * phi_var * rho /
         (sigma2 * phi_var + rho * phi_mean * n_eff_item * pow(xreg[ii][j], 2));
    b_s[ui][j] = (1 - lru) * b_s[ui][j] + lru * gr;
  }
}

HyperParamStats* Regression::create_hyperparam_stats() {
  RegressionHyperParamStats* hps = new RegressionHyperParamStats();
  hps->initialize(sigma2_0, rho_0);
  return hps;
}

void Regression::update_hyperparam(HyperParamStats* stats) {
  update_hyperparam(dynamic_cast<RegressionHyperParamStats*>(stats));
}

void Regression::update_hyperparam(RegressionHyperParamStats* stats) {
  t_hyper += 1;
  double lr = pow(t_hyper, -xi);
  sigma2 = (1 - lr) * sigma2 + lr * stats->get_sigma2_gradient();
  rho = (1 - lr) * rho + lr * stats->get_rho_gradient();
  stats->initialize(sigma2_0, rho_0);
}

double Regression::predict(unsigned long int ui, unsigned long int ii) {
  double out = 0.0;
  for (int j = 0; j < n_component; ++j) out += a_s[ui][j] * xreg[ii][j];
  return out;
}

double Regression::predict(unsigned long int ui, unsigned long int ii,
                           vector<double>& buffer_k) {
  return predict(ui, ii);
}

void Regression::update(unsigned long int ui, unsigned long int ii, double val,
                        vector<double>& buffer_k,
                        RegressionHyperParamStats* stats) {
  double lru = calc_lru(ui);
  double m = predict(ui, ii);
  for (int j = 0; j < n_component; ++j)
    buffer_k[j] = val - m + a_s[ui][j] * xreg[ii][j];
  update_b_s(ui, ii, lru, 1.0, 1.0);
  update_a_s(ui, ii, lru, buffer_k, 1.0, 1.0);

  // TODO Review these updates
  double rho_beta = 0;
  for (int j = 0; j < n_component; ++j)
    rho_beta += pow(a_s[ui][j], 2) + b_s[ui][j];
  double sigma2_beta = pow(val - m, 2);
  stats->update_sigma2_alpha(1);
  stats->update_sigma2_beta(sigma2_beta);
  stats->update_rho_alpha(n_component);
  stats->update_rho_beta(rho_beta);
}

void Regression::update(unsigned long int ui, unsigned long int ii, double val,
                        vector<double>& buffer_k, vector<double>& buffer_n,
                        HyperParamStats* stats) {
  update(ui, ii, val, buffer_k,
         dynamic_cast<RegressionHyperParamStats*>(stats));
}

void Regression::test(unsigned long int ui, unsigned long int ii, double val,
                      test_result& res) {
  double m = predict(ui, ii);
  double pred = m;
  double cond_pred = m;
  double sqerr = pow(val - m, 2);
  double tll = -0.5 * sqerr / sigma2 -
               0.5 * log(2 * boost::math::constants::pi<double>() * sigma2);
  res.pred = pred;
  res.test_loglik = tll;
  res.test_error = sqerr;
  res.cond_pred = cond_pred;
  res.cond_test_loglik = tll;
  res.cond_test_error = sqerr;
}

void Regression::test(unsigned long int ui, unsigned long int ii, double val,
                      vector<double>& buffer_k, vector<double>& buffer_n,
                      test_result& res) {
  test(ui, ii, val, res);
}

void Regression::update_q_n(unsigned long int ui, unsigned long int ii,
                            double log_lambda, double val,
                            vector<double>& phi_mean, vector<double>& phi_var,
                            vector<double>& q_n) {
  double mu = predict(ui, ii);
  for (int n = 1; n < n_trunc; ++n)
    q_n[n - 1] = -0.5 * pow(val - mu * phi_mean[n], 2) / sigma2 / phi_var[n] -
                 0.5 * log(2 * boost::math::constants::pi<double>() * sigma2 *
                           phi_var[n]) -
                 cache_log_fact[n] + n * log_lambda;
  double norm = logsumexp(q_n, n_trunc - 1);
  for (int n = n_trunc - 1; n != 0; --n) q_n[n] = exp(q_n[n - 1] - norm);
  q_n[0] = 0.0;
}

void Regression::update_q_n(unsigned long int ui, unsigned long int ii,
                            double log_lambda, double val,
                            vector<double>& phi_mean, vector<double>& phi_var,
                            vector<double>& buffer_k, vector<double>& q_n) {
  update_q_n(ui, ii, log_lambda, val, phi_mean, phi_var, q_n);
}

void Regression::update(unsigned long int ui, unsigned long int ii, double val,
                        vector<double>& buffer_k,
                        RegressionHyperParamStats* stats, double e_phi_mean,
                        double e_phi_var) {
  double lru = calc_lru(ui);
  double m = predict(ui, ii);
  for (int j = 0; j < n_component; ++j)
    buffer_k[j] = val - e_phi_mean * m + e_phi_mean * a_s[ui][j] * xreg[ii][j];
  update_b_s(ui, ii, lru, e_phi_mean, e_phi_var);
  update_a_s(ui, ii, lru, buffer_k, e_phi_mean, e_phi_var);

  // TODO Review these updates
  double rho_beta = 0;
  for (int j = 0; j < n_component; ++j)
    rho_beta += pow(a_s[ui][j], 2) + b_s[ui][j];
  double sigma2_beta = pow(val, 2) - 2 * val * e_phi_mean * m + pow(m, 2);
  stats->update_sigma2_alpha(1);
  stats->update_sigma2_beta(sigma2_beta);
  stats->update_rho_alpha(n_component);
  stats->update_rho_beta(rho_beta);
}

void Regression::update(unsigned long int ui, unsigned long int ii, double val,
                        vector<double>& buffer_k, vector<double>& buffer_n,
                        HyperParamStats* stats, double e_phi_mean,
                        double e_phi_var) {
  update(ui, ii, val, buffer_k, dynamic_cast<RegressionHyperParamStats*>(stats),
         e_phi_mean, e_phi_var);
}

void Regression::update_test_result(unsigned long int ui, unsigned long int ii,
                                    double log_lambda, double val,
                                    vector<double>& buffer_n,
                                    vector<double>& phi_mean,
                                    vector<double>& phi_var, test_result& res) {
  double mu = predict(ui, ii);
  double element_mean = mu;
  double m = exp(log_lambda);
  double cond_norm = bmath::expm1(m);
  double pred = element_mean * m;
  double cond_pred = element_mean * m * exp(m) / cond_norm;
  double tll = 0.0;
  if (val != 0) {
    for (int n = 1; n < n_trunc; ++n)
      buffer_n[n - 1] =
          -0.5 * pow(val - mu * phi_mean[n], 2) / sigma2 / phi_var[n] -
          0.5 * log(2 * boost::math::constants::pi<double>() * sigma2 *
                    phi_var[n]) -
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

void Regression::update_test_result(unsigned long int ui, unsigned long int ii,
                                    double log_lambda, double val,
                                    vector<double>& buffer_k,
                                    vector<double>& buffer_n,
                                    vector<double>& phi_mean,
                                    vector<double>& phi_var, test_result& res) {
  update_test_result(ui, ii, log_lambda, val, buffer_n, phi_mean, phi_var, res);
}

void Regression::load_xreg(string& fname) {
  H5::H5File file(fname, H5F_ACC_RDONLY);
  H5::DataSet dset = file.openDataSet("xreg");
  H5::DataSpace dspace = dset.getSpace();
  hsize_t dims[2];
  int ndims = dspace.getSimpleExtentDims(dims, NULL);
  int n_xreg_ = dims[1];
  dset.close();
  dspace.close();
  set_n_xreg(n_xreg_);
  string dsetname = "xreg";
  read_h5array2d(file, dsetname, xreg);
  dsetname = "coef";
  read_h5vector(file, dsetname, eta);
  dsetname = "rho";
  vector<double> rhovect(1);
  read_h5vector(file, dsetname, rhovect);
  rho = rhovect[0];
  rho_0 = rho;
  file.close();
}

void Regression::reset_coef() {
  for (size_t i = 0; i < n_user; ++i)
    for (int j = 0; j < n_component; ++j) a_s[i][j] = eta[j];
}
