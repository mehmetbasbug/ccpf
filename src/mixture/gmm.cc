#include "utils.h"
#include "mixture/gmm.h"
using namespace std;
namespace bmath = boost::math;

GMMHyperParamStats::GMMHyperParamStats() {}
void GMMHyperParamStats::initialize(double sigma2_0, double rho_0) {
  // Inverse Gamma prior on rho
  // Inverse Gamma prior on sigma2
  sigma2_alpha = 2;
  sigma2_beta = sigma2_0;
  rho_alpha = 2;
  rho_beta = rho_0;
}
void GMMHyperParamStats::update_sigma2_alpha(double x) {
  sigma2_alpha += x / 2;
}
void GMMHyperParamStats::update_sigma2_beta(double x) { sigma2_beta += x / 2; }
void GMMHyperParamStats::update_rho_alpha(double x) { rho_alpha += x / 2; }
void GMMHyperParamStats::update_rho_beta(double x) { rho_beta += x / 2; }
double GMMHyperParamStats::get_sigma2_gradient() {
  return sigma2_beta / (sigma2_alpha - 1);
}
double GMMHyperParamStats::get_rho_gradient() {
  return rho_beta / (rho_alpha - 1);
}

GMM::GMM() {}
GMM::GMM(unsigned long int dim1_, unsigned long int dim2_, int n_component_,
         double eta_, double rho_, double sigma2_, double xi_, double tau_,
         int n_trunc_) {
  this->initialize(dim1_, dim2_, n_component_, eta_, rho_, sigma2_, xi_, tau_,
                   n_trunc_);
}

void GMM::initialize(unsigned long int dim1_, unsigned long int dim2_,
                     int n_component_, double eta_, double rho_, double sigma2_,
                     double xi_, double tau_, int n_trunc_) {
  dim1 = dim1_;
  dim2 = dim2_;
  n_component = n_component_;
  eta = eta_;
  eta_0 = eta;
  rho = rho_;
  rho_0 = rho;
  sigma2 = sigma2_;
  sigma2_0 = sigma2;
  n_trunc = n_trunc_;
  xi = xi_;
  tau = tau_;
  t_hyper = tau;
  t_learning.resize(dim1_);
  a_s.resize(boost::extents[dim1][n_component]);
  b_s.resize(boost::extents[dim1][n_component]);
  for (size_t i = 0; i < dim1; ++i) {
    t_learning[i] = tau;
    for (int j = 0; j < n_component; ++j) {
      a_s[i][j] = eta;
      b_s[i][j] = rho;
    }
  }
  cache_log_fact.resize(n_trunc);
  for (int j = 0; j < n_trunc; ++j) cache_log_fact[j] = bmath::lgamma(j + 1);
}

double GMM::get_eta() { return eta; }

double GMM::get_eta_0() { return eta_0; }

double GMM::get_rho() { return rho; }

double GMM::get_rho_0() { return rho_0; }

double GMM::get_sigma2() { return sigma2; }

double GMM::get_sigma2_0() { return sigma2_0; }

int GMM::get_n_trunc() { return n_trunc; }

int GMM::get_n_component() { return n_component; }

void GMM::set_eta(double eta_) { eta = eta_; }

void GMM::set_eta_0(double eta_0_) { eta_0 = eta_0_; }

void GMM::set_rho(double rho_) { rho = rho_; }

void GMM::set_rho_0(double rho_0_) { rho_0 = rho_0_; }

void GMM::set_sigma2(double sigma2_) { sigma2 = sigma2_; }

void GMM::set_sigma2_0(double sigma2_0_) { sigma2_0 = sigma2_0_; }

void GMM::serialize(string& fname) {
  H5::H5File file(fname, H5F_ACC_TRUNC);
  vector<unsigned long int> uiparams(3);
  uiparams[0] = dim1;
  uiparams[1] = dim2;
  uiparams[2] = t_hyper;
  string dsetname = "uiparams";
  write_h5vector(file, dsetname, uiparams);

  vector<int> iparams(2);
  iparams[0] = n_component;
  iparams[1] = n_trunc;
  dsetname = "iparams";
  write_h5vector(file, dsetname, iparams);

  vector<double> params(8);
  params[0] = eta;
  params[1] = eta_0;
  params[2] = rho;
  params[3] = rho_0;
  params[4] = sigma2;
  params[5] = sigma2_0;
  params[6] = xi;
  params[7] = tau;
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

void GMM::deserialize(string& fname) {
  H5::H5File file(fname, H5F_ACC_RDONLY);
  vector<unsigned long int> uiparams(3);
  string dsetname = "uiparams";
  read_h5vector(file, dsetname, uiparams);
  unsigned long int dim1_ = uiparams[0];
  unsigned long int dim2_ = uiparams[1];
  unsigned long int t_hyper_ = uiparams[1];

  vector<int> iparams(2);
  dsetname = "iparams";
  read_h5vector(file, dsetname, iparams);
  int n_component_ = iparams[0];
  int n_trunc_ = iparams[1];

  vector<double> params(8);
  dsetname = "params";
  read_h5vector(file, dsetname, params);
  double eta_ = params[0];
  double eta_0_ = params[1];
  double rho_ = params[2];
  double rho_0_ = params[3];
  double sigma2_ = params[4];
  double sigma2_0_ = params[5];
  double xi_ = params[6];
  double tau_ = params[7];
  this->initialize(dim1_, dim2_, n_component_, eta_, rho_, sigma2_, xi_, tau_,
                   n_trunc_);
  dsetname = "a_s";
  read_h5array2d(file, dsetname, a_s);
  dsetname = "b_s";
  read_h5array2d(file, dsetname, b_s);
  dsetname = "t_learning";
  read_h5vector(file, dsetname, t_learning);
  eta_0 = eta_0_;
  rho_0 = rho_0_;
  sigma2_0 = sigma2_0_;
  t_hyper = t_hyper_;
  file.close();
}

double GMM::calc_lr(unsigned long int ind) {
  t_learning[ind] += 1;
  return pow(t_learning[ind], -xi);
}

void GMM::update_a_s(unsigned long int ind, double lr, double err,
                     double phi_mean = 1.0, double phi_var = 1.0) {
  double gr = 0.0;
  for (int j = 0; j < n_component; ++j) {
    gr = (eta * phi_var * sigma2 +
          dim2 * rho * phi_mean * (err + phi_mean * a_s[ind][j])) /
         (sigma2 * phi_var + dim2 * rho * phi_mean);
    a_s[ind][j] = (1 - lr) * a_s[ind][j] + lr * gr;
  }
}

void GMM::update_b_s(unsigned long int ind, double lr, double phi_mean = 1.0,
                     double phi_var = 1.0) {
  double gr = 0.0;
  for (int j = 0; j < n_component; ++j) {
    gr = (rho * sigma2 * phi_var) / (sigma2 * phi_var + dim2 * rho * phi_mean);
    b_s[ind][j] = (1 - lr) * b_s[ind][j] + lr * gr;
  }
}

HyperParamStats* GMM::create_hyperparam_stats() {
  GMMHyperParamStats* hps = new GMMHyperParamStats();
  hps->initialize(sigma2_0, rho_0);
  return hps;
}

void GMM::update_hyperparam(GMMHyperParamStats* stats) {
  t_hyper += 1;
  double lr = pow(t_hyper, -xi);
  rho = (1 - lr) * rho + lr * stats->get_rho_gradient();
  sigma2 = (1 - lr) * sigma2 + lr * stats->get_sigma2_gradient();
  stats->initialize(sigma2_0, rho_0);
}

double GMM::predict(unsigned long int ind) {
  double out = 0.0;
  for (int j = 0; j < n_component; ++j) out += a_s[ind][j];
  return out;
}

void GMM::update(unsigned long int ind, double val, GMMHyperParamStats* stats) {
  double lr = calc_lr(ind);
  double err = val - predict(ind);
  update_b_s(ind, lr, 1.0, 1.0);
  update_a_s(ind, lr, err, 1.0, 1.0);

  // TODO Review these updates
  double rho_beta = 0;
  for (int j = 0; j < n_component; ++j)
    rho_beta += pow(a_s[ind][j], 2) + b_s[ind][j];
  double sigma2_beta = pow(err, 2);
  stats->update_sigma2_alpha(1);
  stats->update_sigma2_beta(sigma2_beta);
  stats->update_rho_alpha(n_component);
  stats->update_rho_beta(rho_beta);
}

void GMM::test(unsigned long int ind, double val, test_result& res) {
  double mean = this->predict(ind);
  double pred = mean;
  double cond_pred = mean;
  double err2 = pow(val - mean, 2);
  double tll = -0.5 * err2 / sigma2 - 0.5 * log(2 * boost::math::constants::pi<double>() * sigma2);
  res.pred = pred;
  res.test_loglik = tll;
  res.test_error = err2;
  res.cond_pred = cond_pred;
  res.cond_test_loglik = tll;
  res.cond_test_error = err2;
}

void GMM::update_q_n(unsigned long int ind, double log_lambda, double val,
                     vector<double>& phi_mean, vector<double>& phi_var,
                     vector<double>& q_n) {
  double mu = predict(ind);
  for (int n = 1; n < n_trunc; ++n)
    q_n[n - 1] = -0.5 * pow(val - mu * phi_mean[n], 2) / sigma2 / phi_var[n] -
                 0.5 * log(2 * boost::math::constants::pi<double>() * sigma2 * phi_var[n]) - cache_log_fact[n] +
                 n * log_lambda;
  double norm = logsumexp(q_n, n_trunc - 1);
  for (int n = n_trunc - 1; n !=0; --n) q_n[n] = exp(q_n[n-1] - norm);
  q_n[0] = 0.0;
}

void GMM::update(unsigned long int ind, double val, GMMHyperParamStats* stats,
                 double e_phi_mean, double e_phi_var) {
  double lr = calc_lr(ind);
  double err = val - predict(ind) * e_phi_mean;
  update_b_s(ind, lr, e_phi_mean, e_phi_var);
  update_a_s(ind, lr, err, e_phi_mean, e_phi_var);

  // TODO Review these updates
  double rho_beta = 0;
  for (int j = 0; j < n_component; ++j)
    rho_beta += pow(a_s[ind][j], 2) + b_s[ind][j];
  double sigma2_beta = pow(err, 2) / e_phi_var;
  stats->update_sigma2_alpha(1);
  stats->update_sigma2_beta(sigma2_beta);
  stats->update_rho_alpha(n_component);
  stats->update_rho_beta(rho_beta);
}

void GMM::update_test_result(unsigned long int ind, double log_lambda,
                             double val, vector<double>& buffer_n,
                             vector<double>& phi_mean, vector<double>& phi_var,
                             test_result& res) {
  double mu = predict(ind);
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
          0.5 * log(2 * boost::math::constants::pi<double>() * sigma2 * phi_var[n]) - cache_log_fact[n] +
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

UserGMM::UserGMM() {}
UserGMM::UserGMM(unsigned long int n_user_, unsigned long int n_item_,
                 int n_component_, double eta_, double rho_, double sigma2_,
                 double xi_, double tau_, int n_trunc_)
    : GMM(n_user_, n_item_, n_component_, eta_, rho_, sigma2_, xi_, tau_,
          n_trunc_) {}

void UserGMM::initialize(unsigned long int n_user_, unsigned long int n_item_,
                         int n_component_, double eta_, double rho_,
                         double sigma2_, double xi_, double tau_,
                         int n_trunc_) {
  this->GMM::initialize(n_user_, n_item_, n_component_, eta_, rho_, sigma2_,
                        xi_, tau_, n_trunc_);
}

void UserGMM::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                            int n_trunc_, double xi_ = 0.7,
                                            double tau_ = 10000) {
  const triplets& ind2train = sm.get_ind2train();
  normal_params params = mle_normal(ind2train, 1000);
  double mu_ = params.mu;
  double sigma2_ = params.sigma2;
  double eta_ = mu_ / (double)n_component_;
  double rho_ = sigma2_ / (double)n_component_;
  unsigned long int n_item_eff = sm.get_n_item() * (1 - sm.get_sparsity());
  this->GMM::initialize(sm.get_n_user(), n_item_eff, n_component_, eta_, rho_,
                        sigma2_, xi_, tau_, n_trunc_);
}

void UserGMM::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                            double xi_ = 0.7,
                                            double tau_ = 10000) {
  initialize_with_sparse_matrix(sm, n_component_, 100, xi_, tau_);
}

void UserGMM::serialize(string& fname) { GMM::serialize(fname); }

void UserGMM::deserialize(string& fname) { GMM::deserialize(fname); }

int UserGMM::get_n_trunc() { return GMM::get_n_trunc(); }

int UserGMM::get_n_component() { return GMM::get_n_component(); }

double UserGMM::get_sigma2() { return GMM::get_sigma2(); }

HyperParamStats* UserGMM::create_hyperparam_stats() {
  return GMM::create_hyperparam_stats();
}

void UserGMM::update_hyperparam(HyperParamStats* stats) {
  return GMM::update_hyperparam(dynamic_cast<GMMHyperParamStats*>(stats));
}

double UserGMM::predict(unsigned long int ui, unsigned long int ii,
                        vector<double>& buffer_k) {
  return GMM::predict(ui);
}

void UserGMM::update(unsigned long int ui, unsigned long int ii, double val,
                     vector<double>& buffer_k, vector<double>& buffer_n,
                     HyperParamStats* stats) {
  GMM::update(ui, val, dynamic_cast<GMMHyperParamStats*>(stats));
}

void UserGMM::test(unsigned long int ui, unsigned long int ii, double val,
                   vector<double>& buffer_k, vector<double>& buffer_n,
                   test_result& res) {
  GMM::test(ui, val, res);
}

void UserGMM::update_q_n(unsigned long int ui, unsigned long int ii,
                         double log_lambda, double val,
                         vector<double>& phi_mean, vector<double>& phi_var,
                         vector<double>& buffer_k, vector<double>& q_n) {
  GMM::update_q_n(ui, log_lambda, val, phi_mean, phi_var, q_n);
}

void UserGMM::update(unsigned long int ui, unsigned long int ii, double val,
                     vector<double>& buffer_k, vector<double>& buffer_n,
                     HyperParamStats* stats, double e_phi_mean,
                     double e_phi_var) {
  GMM::update(ui, val, dynamic_cast<GMMHyperParamStats*>(stats), e_phi_mean,
              e_phi_var);
}

void UserGMM::update_test_result(unsigned long int ui, unsigned long int ii,
                                 double log_lambda, double val,
                                 vector<double>& buffer_k,
                                 vector<double>& buffer_n,
                                 vector<double>& phi_mean,
                                 vector<double>& phi_var, test_result& res) {
  GMM::update_test_result(ui, log_lambda, val, buffer_n, phi_mean, phi_var,
                          res);
}

ItemGMM::ItemGMM() {}
ItemGMM::ItemGMM(unsigned long int n_user_, unsigned long int n_item_,
                 int n_component_, double eta_, double rho_, double sigma2_,
                 double xi_, double tau_, int n_trunc_)
    : GMM(n_item_, n_user_, n_component_, eta_, rho_, sigma2_, xi_, tau_,
          n_trunc_) {}

void ItemGMM::initialize(unsigned long int n_user_, unsigned long int n_item_,
                         int n_component_, double eta_, double rho_,
                         double sigma2_, double xi_, double tau_,
                         int n_trunc_) {
  this->GMM::initialize(n_item_, n_user_, n_component_, eta_, rho_, sigma2_,
                        xi_, tau_, n_trunc_);
}

void ItemGMM::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                            int n_trunc_, double xi_ = 0.7,
                                            double tau_ = 10000) {
  const triplets& ind2train = sm.get_ind2train();
  normal_params params = mle_normal(ind2train, 1000);
  double mu_ = params.mu;
  double sigma2_ = params.sigma2;
  double eta_ = mu_ / (double)n_component_;
  double rho_ = sigma2_ / (double)n_component_;
  unsigned long int n_user_eff = sm.get_n_user() * (1 - sm.get_sparsity());
  this->GMM::initialize(sm.get_n_item(), n_user_eff, n_component_, eta_, rho_,
                        sigma2_, xi_, tau_, n_trunc_);
}

void ItemGMM::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                            double xi_ = 0.7,
                                            double tau_ = 10000) {
  initialize_with_sparse_matrix(sm, n_component_, 100, xi_, tau_);
}

void ItemGMM::serialize(string& fname) { GMM::serialize(fname); }

void ItemGMM::deserialize(string& fname) { GMM::deserialize(fname); }

int ItemGMM::get_n_trunc() { return GMM::get_n_trunc(); }

int ItemGMM::get_n_component() { return GMM::get_n_component(); }

double ItemGMM::get_sigma2() { return GMM::get_sigma2(); }

HyperParamStats* ItemGMM::create_hyperparam_stats() {
  return GMM::create_hyperparam_stats();
}

void ItemGMM::update_hyperparam(HyperParamStats* stats) {
  return GMM::update_hyperparam(dynamic_cast<GMMHyperParamStats*>(stats));
}

double ItemGMM::predict(unsigned long int ui, unsigned long int ii,
                        vector<double>& buffer_k) {
  return GMM::predict(ii);
}

void ItemGMM::update(unsigned long int ui, unsigned long int ii, double val,
                     vector<double>& buffer_k, vector<double>& buffer_n,
                     HyperParamStats* stats) {
  GMM::update(ii, val, dynamic_cast<GMMHyperParamStats*>(stats));
}

void ItemGMM::test(unsigned long int ui, unsigned long int ii, double val,
                   vector<double>& buffer_k, vector<double>& buffer_n,
                   test_result& res) {
  GMM::test(ii, val, res);
}

void ItemGMM::update_q_n(unsigned long int ui, unsigned long int ii,
                         double log_lambda, double val,
                         vector<double>& phi_mean, vector<double>& phi_var,
                         vector<double>& buffer_k, vector<double>& q_n) {
  GMM::update_q_n(ii, log_lambda, val, phi_mean, phi_var, q_n);
}

void ItemGMM::update(unsigned long int ui, unsigned long int ii, double val,
                     vector<double>& buffer_k, vector<double>& buffer_n,
                     HyperParamStats* stats, double e_phi_mean,
                     double e_phi_var) {
  GMM::update(ii, val, dynamic_cast<GMMHyperParamStats*>(stats), e_phi_mean,
              e_phi_var);
}

void ItemGMM::update_test_result(unsigned long int ui, unsigned long int ii,
                                 double log_lambda, double val,
                                 vector<double>& buffer_k,
                                 vector<double>& buffer_n,
                                 vector<double>& phi_mean,
                                 vector<double>& phi_var, test_result& res) {
  GMM::update_test_result(ii, log_lambda, val, buffer_n, phi_mean, phi_var,
                          res);
}
