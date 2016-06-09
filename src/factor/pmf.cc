#include "utils.h"
#include "factor/pmf.h"
using namespace std;
namespace bmath = boost::math;

PMFHyperParamStats::PMFHyperParamStats() {}
void PMFHyperParamStats::initialize(double sigma2_0, double rho_0,
                                    double omega_0) {
  // Inverse Gamma prior on variance
  sigma2_alpha = 2;
  sigma2_beta = sigma2_0;
  rho_alpha = 2;
  rho_beta = rho_0;
  omega_alpha = 2;
  omega_beta = omega_0;
}
void PMFHyperParamStats::update_rho_alpha(double x) { rho_alpha += x / 2; }
void PMFHyperParamStats::update_rho_beta(double x) { rho_beta += x / 2; }
void PMFHyperParamStats::update_omega_alpha(double x) { omega_alpha += x / 2; }
void PMFHyperParamStats::update_omega_beta(double x) { omega_beta += x / 2; }
void PMFHyperParamStats::update_sigma2_alpha(double x) {
  sigma2_alpha += x / 2;
}
void PMFHyperParamStats::update_sigma2_beta(double x) { sigma2_beta += x / 2; }
double PMFHyperParamStats::get_rho_gradient() {
  return rho_beta / (rho_alpha - 1);
}
double PMFHyperParamStats::get_omega_gradient() {
  return omega_beta / (omega_alpha - 1);
}
double PMFHyperParamStats::get_sigma2_gradient() {
  return sigma2_beta / (sigma2_alpha - 1);
}

PMF::PMF() {}
PMF::PMF(unsigned long int n_user_, unsigned long int n_item_, int n_component_,
         double sparsity_, double eta_, double rho_, double zeta_,
         double omega_, double sigma2_, double xi_, double tau_, int n_trunc_) {
  this->initialize(n_user_, n_item_, n_component_, sparsity_, eta_, rho_, zeta_,
                   omega_, sigma2_, xi_, tau_, n_trunc_);
}

void PMF::initialize(unsigned long int n_user_, unsigned long int n_item_,
                     int n_component_, double sparsity_, double eta_,
                     double rho_, double zeta_, double omega_, double sigma2_,
                     double xi_, double tau_, int n_trunc_) {
  n_user = n_user_;
  sparsity = sparsity_;
  n_eff_user = (1 - sparsity) * n_user;
  n_item = n_item_;
  n_eff_item = (1 - sparsity) * n_item;
  n_component = n_component_;
  eta = eta_;
  eta_0 = eta;
  rho = rho_;
  rho_0 = rho;
  zeta = zeta_;
  zeta_0 = zeta;
  omega = omega_;
  omega_0 = omega;
  sigma2 = sigma2_;
  sigma2_0 = sigma2;
  n_trunc = n_trunc_;
  xi = xi_;
  tau = tau_;
  t_hyper = tau;
  t_user.resize(n_user_);
  t_item.resize(n_item_);
  a_s.resize(boost::extents[n_user][n_component]);
  b_s.resize(boost::extents[n_user][n_component]);
  a_v.resize(boost::extents[n_item][n_component]);
  b_v.resize(boost::extents[n_item][n_component]);
  for (size_t i = 0; i < n_user; ++i) {
    t_user[i] = tau;
    for (int j = 0; j < n_component; ++j) {
      a_s[i][j] = eta * (1.0 + 0.1 * ((double)rand() / RAND_MAX));
      b_s[i][j] = rho;
    }
  }
  for (size_t i = 0; i < n_item; ++i) {
    t_item[i] = tau;
    for (int j = 0; j < n_component; ++j) {
      a_v[i][j] = zeta * (1.0 + 0.1 * ((double)rand() / RAND_MAX));
      b_v[i][j] = omega;
    }
  }
  cache_log_fact.resize(n_trunc);
  for (int j = 0; j < n_trunc; ++j) cache_log_fact[j] = bmath::lgamma(j + 1);
}

void PMF::initialize_with_sparse_matrix(SparseMatrix& sm,
                                        int n_component_,
                                        int n_trunc_,
                                        double xi_ = 0.7,
                                        double tau_ = 10000) {
  const triplets& ind2train = sm.get_ind2train();
  normal_params params = mle_normal(ind2train, 1000);
  double mu_ = params.mu;
  double sigma2_ = params.sigma2;
  double mu_k = mu_ / (double)n_component_;
  double eta_ = sqrt(mu_k);
  double zeta_ = eta_;
  double sigma2_k = sigma2_ / (double)n_component_;
  double rho_ = abs(sqrt(sigma2_k + mu_k * mu_k) - mu_k);
  double omega_ = rho_;
  this->initialize(sm.get_n_user(), sm.get_n_item(), n_component_,
                   sm.get_sparsity(), eta_, rho_, zeta_, omega_,
                   sigma2_, xi_, tau_, n_trunc_);
}

void PMF::initialize_with_sparse_matrix(SparseMatrix& sm,
                                        int n_component_,
                                        double xi_ = 0.7,
                                        double tau_ = 10000){
  initialize_with_sparse_matrix(sm, n_component_, 10, xi_, tau_);
}

void PMF::serialize(string& fname) {
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

  vector<double> params(13);
  params[0] = eta;
  params[1] = eta_0;
  params[2] = rho;
  params[3] = rho_0;
  params[4] = zeta;
  params[5] = zeta_0;
  params[6] = omega;
  params[7] = omega_0;
  params[8] = sigma2;
  params[9] = sigma2_0;
  params[10] = xi;
  params[11] = tau;
  params[12] = sparsity;
  dsetname = "params";
  write_h5vector(file, dsetname, params);
  dsetname = "a_s";
  write_h5array2d(file, dsetname, a_s);
  dsetname = "b_s";
  write_h5array2d(file, dsetname, b_s);
  dsetname = "t_user";
  write_h5vector(file, dsetname, t_user);
  dsetname = "a_v";
  write_h5array2d(file, dsetname, a_v);
  dsetname = "b_v";
  write_h5array2d(file, dsetname, b_v);
  dsetname = "t_item";
  write_h5vector(file, dsetname, t_item);
  file.close();
}

void PMF::deserialize(string& fname) {
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

  vector<double> params(13);
  dsetname = "params";
  read_h5vector(file, dsetname, params);
  double eta_ = params[0];
  double eta_0_ = params[1];
  double rho_ = params[2];
  double rho_0_ = params[3];
  double zeta_ = params[4];
  double zeta_0_ = params[5];
  double omega_ = params[6];
  double omega_0_ = params[7];
  double sigma2_ = params[8];
  double sigma2_0_ = params[9];
  double xi_ = params[10];
  double tau_ = params[11];
  double sparsity_ = params[12];
  this->initialize(n_user_, n_item_, n_component_, sparsity_, eta_, rho_, zeta_,
                   omega_, sigma2_, xi_, tau_, n_trunc_);
  dsetname = "a_s";
  read_h5array2d(file, dsetname, a_s);
  dsetname = "b_s";
  read_h5array2d(file, dsetname, b_s);
  dsetname = "t_user";
  read_h5vector(file, dsetname, t_user);
  dsetname = "a_v";
  read_h5array2d(file, dsetname, a_v);
  dsetname = "b_v";
  read_h5array2d(file, dsetname, b_v);
  dsetname = "t_item";
  read_h5vector(file, dsetname, t_item);
  eta_0 = eta_0_;
  rho_0 = rho_0_;
  zeta_0 = zeta_0_;
  omega_0 = omega_0_;
  sigma2_0 = sigma2_0_;
  t_hyper = t_hyper_;
  file.close();
}

double PMF::get_eta() { return eta; }

double PMF::get_eta_0() { return eta_0; }

double PMF::get_rho() { return rho; }

double PMF::get_rho_0() { return rho_0; }

double PMF::get_zeta() { return zeta; }

double PMF::get_zeta_0() { return zeta_0; }

double PMF::get_omega() { return omega; }

double PMF::get_omega_0() { return omega_0; }

double PMF::get_sigma2() { return sigma2; }

double PMF::get_sigma2_0() { return sigma2_0; }

int PMF::get_n_trunc() { return n_trunc; }

int PMF::get_n_component() { return n_component; }

void PMF::set_eta(double eta_) { eta = eta_; }

void PMF::set_eta_0(double eta_0_) { eta_0 = eta_0_; }

void PMF::set_rho(double rho_) { rho = rho_; }

void PMF::set_rho_0(double rho_0_) { rho_0 = rho_0_; }

void PMF::set_zeta(double zeta_) { zeta = zeta_; }

void PMF::set_zeta_0(double zeta_0_) { zeta_0 = zeta_0_; }

void PMF::set_omega(double omega_) { omega = omega_; }

void PMF::set_omega_0(double omega_0_) { omega_0 = omega_0_; }

void PMF::set_sigma2(double sigma2_) { sigma2 = sigma2_; }

void PMF::set_sigma2_0(double sigma2_0_) { sigma2_0 = sigma2_0_; }

double PMF::calc_lru(unsigned long int ui) {
  t_user[ui] += 1;
  return pow(t_user[ui], -xi);
}

double PMF::calc_lri(unsigned long int ii) {
  t_item[ii] += 1;
  return pow(t_item[ii], -xi);
}

void PMF::update_a_s(unsigned long int ui, unsigned long int ii, double lru,
                     vector<double>& err, double phi_mean = 1.0,
                     double phi_var = 1.0) {
  double gr = 0.0;
  for (int j = 0; j < n_component; ++j) {
    gr = (eta * phi_var *sigma2 + rho * phi_mean * n_eff_item * err[j] * a_v[ii][j]) /
         (sigma2 * phi_var + rho * phi_mean * n_eff_item * (pow(a_v[ii][j], 2) + b_v[ii][j]));
    a_s[ui][j] = (1 - lru) * a_s[ui][j] + lru * gr;
  }
}

void PMF::update_b_s(unsigned long int ui, unsigned long int ii, double lru,
                     double phi_mean = 1.0, double phi_var = 1.0) {
  double gr = 0.0;
  for (int j = 0; j < n_component; ++j) {
    gr = sigma2 * phi_var * rho /
         (sigma2 * phi_var + rho * phi_mean * n_eff_item * (pow(a_v[ii][j], 2) + b_v[ii][j]));
    b_s[ui][j] = (1 - lru) * b_s[ui][j] + lru * gr;
  }
}

void PMF::update_a_v(unsigned long int ui, unsigned long int ii, double lri,
                     vector<double>& err, double phi_mean = 1.0,
                     double phi_var = 1.0) {
  double gr = 0.0;
  for (int j = 0; j < n_component; ++j) {
    gr = (zeta * phi_var *sigma2 + omega * phi_mean * n_eff_user * err[j] * a_s[ui][j]) /
         (sigma2 * phi_var + omega * phi_mean * n_eff_user * (pow(a_s[ui][j], 2) + b_s[ui][j]));
    a_v[ii][j] = (1 - lri) * a_v[ii][j] + lri * gr;
  }
}

void PMF::update_b_v(unsigned long int ui, unsigned long int ii, double lri,
                     double phi_mean = 1.0, double phi_var = 1.0) {
  double gr = 0.0;
  for (int j = 0; j < n_component; ++j) {
    gr = sigma2 * phi_var * omega /
         (sigma2 * phi_var + omega * phi_mean * n_eff_user * (pow(a_s[ui][j], 2) + b_s[ui][j]));
    b_v[ii][j] = (1 - lri) * b_v[ii][j] + lri * gr;
  }
}

HyperParamStats* PMF::create_hyperparam_stats() {
  PMFHyperParamStats* hps = new PMFHyperParamStats();
  hps->initialize(sigma2_0, rho_0, omega_0);
  return hps;
}

void PMF::update_hyperparam(PMFHyperParamStats* stats) {
  t_hyper += 1;
  double lr = pow(t_hyper, -xi);
  rho = (1 - lr) * rho + lr * stats->get_rho_gradient();
  omega = (1 - lr) * omega + lr * stats->get_omega_gradient();
  sigma2 = (1 - lr) * sigma2 + lr * stats->get_sigma2_gradient();
  stats->initialize(sigma2_0, rho_0, omega_0);
}

void PMF::update_hyperparam(HyperParamStats* stats) {
  update_hyperparam(dynamic_cast<PMFHyperParamStats*>(stats));
}

double PMF::predict(unsigned long int ui, unsigned long int ii) {
  double out = 0.0;
  for (int j = 0; j < n_component; ++j)
    out += a_s[ui][j] * a_v[ii][j];
  return out;
}

double PMF::predict(unsigned long int ui, unsigned long int ii,
                    vector<double>& buffer_k) {
  return predict(ui, ii);
}

void PMF::update(unsigned long int ui, unsigned long int ii, double val,
                 vector<double>& buffer_k, PMFHyperParamStats* stats) {
  double lru = calc_lru(ui);
  double lri = calc_lri(ii);
  double m = predict(ui, ii);
  for (int j = 0; j < n_component; ++j)
    buffer_k[j] = val - m + a_s[ui][j] * a_v[ii][j];
  update_b_s(ui, ii, lru, 1.0, 1.0);
  update_a_s(ui, ii, lru, buffer_k, 1.0, 1.0);
  update_b_v(ui, ii, lri, 1.0, 1.0);
  update_a_v(ii, ii, lri, buffer_k, 1.0, 1.0);

  // TODO Review these updates
  double rho_beta = 0;
  double omega_beta = 0;
  for (int j = 0; j < n_component; ++j) {
    rho_beta += pow(a_s[ui][j], 2) + b_s[ui][j];
    omega_beta += pow(a_v[ii][j], 2) + b_v[ii][j];
  }
  double sigma2_beta = pow(val, 2) - 2 * val * m + pow(m, 2);
  stats->update_sigma2_alpha(1);
  stats->update_sigma2_beta(sigma2_beta);
  stats->update_rho_alpha(n_component);
  stats->update_rho_beta(rho_beta);
  stats->update_omega_alpha(n_component);
  stats->update_omega_beta(omega_beta);
}

void PMF::update(unsigned long int ui, unsigned long int ii, double val,
                 vector<double>& buffer_k, vector<double>& buffer_n,
                 HyperParamStats* stats) {
  update(ui, ii, val, buffer_k, dynamic_cast<PMFHyperParamStats*>(stats));
}

void PMF::test(unsigned long int ui, unsigned long int ii, double val,
               test_result& res) {
  double m = predict(ui, ii);
  double pred = m;
  double cond_pred = m;
  double sqerr = pow(val - m, 2);
  double tll = -0.5 * sqerr / sigma2 - 0.5 * log(2 * boost::math::constants::pi<double>() * sigma2);
  res.pred = pred;
  res.test_loglik = tll;
  res.test_error = sqerr;
  res.cond_pred = cond_pred;
  res.cond_test_loglik = tll;
  res.cond_test_error = sqerr;
}

void PMF::test(unsigned long int ui, unsigned long int ii, double val,
               vector<double>& buffer_k, vector<double>& buffer_n,
               test_result& res) {
  test(ui, ii, val, res);
}

void PMF::update_q_n(unsigned long int ui,
                     unsigned long int ii,
                     double log_lambda,
                     double val,
                     vector<double>& phi_mean,
                     vector<double>& phi_var,
                     vector<double>& q_n) {
  double mu = predict(ui, ii);
  for (int n = 1; n < n_trunc; ++n)
    q_n[n - 1] = -0.5 * pow(val - mu * phi_mean[n], 2) / sigma2 / phi_var[n] -
                 0.5 * log(2 * boost::math::constants::pi<double>() * sigma2 * phi_var[n]) - cache_log_fact[n] +
                 n * log_lambda;
  double norm = logsumexp(q_n, n_trunc - 1);
  for (int n = n_trunc - 1; n !=0; --n) q_n[n] = exp(q_n[n-1] - norm);
  q_n[0] = 0.0;
}

void PMF::update_q_n(unsigned long int ui,
                     unsigned long int ii,
                     double log_lambda,
                     double val,
                     vector<double>& phi_mean,
                     vector<double>& phi_var,
                     vector<double>& buffer_k,
                     vector<double>& q_n) {
  update_q_n(ui, ii, log_lambda, val, phi_mean, phi_var, q_n);
}

void PMF::update(unsigned long int ui, unsigned long int ii, double val,
                 vector<double>& buffer_k, PMFHyperParamStats* stats,
                 double e_phi_mean, double e_phi_var) {
  double lru = calc_lru(ui);
  double lri = calc_lri(ii);
  double m = predict(ui, ii);
  for (int j = 0; j < n_component; ++j)
    buffer_k[j] = val - e_phi_mean * m + e_phi_mean * a_s[ui][j] * a_v[ii][j];
  update_b_s(ui, ii, lru, e_phi_mean, e_phi_var);
  update_a_s(ui, ii, lru, buffer_k, e_phi_mean, e_phi_var);
  update_b_v(ui, ii, lri, e_phi_mean, e_phi_var);
  update_a_v(ii, ii, lri, buffer_k, e_phi_mean, e_phi_var);

  // TODO Review these updates
  double rho_beta = 0;
  double omega_beta = 0;
  for (int j = 0; j < n_component; ++j) {
    rho_beta += pow(a_s[ui][j], 2) + b_s[ui][j];
    omega_beta += pow(a_v[ii][j], 2) + b_v[ii][j];
  }
  double sigma2_beta = pow(val, 2) - 2 * val * e_phi_mean* m + pow(m, 2);
  stats->update_sigma2_alpha(1);
  stats->update_sigma2_beta(sigma2_beta);
  stats->update_rho_alpha(n_component);
  stats->update_rho_beta(rho_beta);
  stats->update_omega_alpha(n_component);
  stats->update_omega_beta(omega_beta);
}

void PMF::update(unsigned long int ui, unsigned long int ii, double val,
                 vector<double>& buffer_k, vector<double>& buffer_n,
                 HyperParamStats* stats, double e_phi_mean, double e_phi_var) {
  update(ui, ii, val, buffer_k, dynamic_cast<PMFHyperParamStats*>(stats),
         e_phi_mean, e_phi_var);
}

void PMF::update_test_result(unsigned long int ui,
                             unsigned long int ii,
                             double log_lambda,
                             double val,
                             vector<double>& buffer_n,
                             vector<double>& phi_mean,
                             vector<double>& phi_var,
                             test_result& res) {
  double mu = this->predict(ui,ii);
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

void PMF::update_test_result(unsigned long int ui,
                             unsigned long int ii,
                             double log_lambda,
                             double val,
                             vector<double>& buffer_k,
                             vector<double>& buffer_n,
                             vector<double>& phi_mean,
                             vector<double>& phi_var,
                             test_result& res) {
  update_test_result(ui, ii, log_lambda, val, buffer_n, phi_mean, phi_var, res);
}
