#include "utils.h"
#include "factor/hpf.h"
using namespace std;
namespace bmath = boost::math;

HPFBase::HPFBase() {}
HPFBase::HPFBase(unsigned long int n_user_, unsigned long int n_item_,
                 int n_component_, double sparsity_, double eta_, double rho_,
                 double varrho_, double zeta_, double omega_, double varpi_,
                 double xi_, double tau_, int n_trunc_, int val_max_) {
  this->initialize(n_user_, n_item_, n_component_, sparsity_, eta_, rho_,
                   varrho_, zeta_, omega_, varpi_, xi_, tau_, n_trunc_,
                   val_max_);
}

void HPFBase::initialize(unsigned long int n_user_, unsigned long int n_item_,
                         int n_component_, double sparsity_, double eta_,
                         double rho_, double varrho_, double zeta_,
                         double omega_, double varpi_, double xi_, double tau_,
                         int n_trunc_, int val_max_) {
  n_user = n_user_;
  sparsity = sparsity_;
  n_eff_user = (1 - sparsity) * n_user;
  n_item = n_item_;
  n_eff_item = (1 - sparsity) * n_item;
  n_component = n_component_;
  n_trunc = n_trunc_;
  val_max = val_max_;
  eta = eta_;
  eta_0 = eta;
  rho = rho_;
  rho_0 = rho;
  varrho = varrho_;
  varrho_0 = varrho;
  zeta = zeta_;
  zeta_0 = zeta;
  omega = omega_;
  omega_0 = omega;
  varpi = varpi_;
  varpi_0 = varpi;
  xi = xi_;
  tau = tau_;
  t_hyper = tau;
  t_user.resize(n_user_);
  t_item.resize(n_item_);
  a_r.resize(n_user_);
  b_r.resize(n_user_);
  a_s.resize(boost::extents[n_user][n_component]);
  b_s.resize(boost::extents[n_user][n_component]);
  a_v.resize(boost::extents[n_item][n_component]);
  b_v.resize(boost::extents[n_item][n_component]);
  a_w.resize(n_item_);
  b_w.resize(n_item_);
  for (int i = 0; i < n_user; ++i) {
    t_user[i] = tau;
    a_r[i] = rho;
    b_r[i] = rho / varrho;
    for (int j = 0; j < n_component; ++j) {
      a_s[i][j] = eta;
      b_s[i][j] = a_r[i] / b_r[i];
    }
  }
  for (int i = 0; i < n_item; ++i) {
    t_item[i] = tau;
    a_w[i] = omega;
    b_w[i] = omega / varpi;
    for (int j = 0; j < n_component; ++j) {
      a_v[i][j] = zeta;
      b_v[i][j] = a_w[i] / b_w[i];
    }
  }
  int max_dim = max(val_max, n_trunc);
  cache_log_fact.resize(max_dim);
  for (int j = 0; j < max_dim; ++j) cache_log_fact[j] = bmath::lgamma(j + 1);
}

void HPFBase::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                            int n_trunc_, double xi_ = 0.7,
                                            double tau_ = 10000) {
  const triplets& ind2train = sm.get_ind2train();
  poisson_params params = mle_poisson(ind2train, 1000);
  double e_m_sq = sqrt(params.lambda / (double)n_component_);
  double varrho_ = 0.1;
  double eta_ = varrho_ * e_m_sq;
  double varpi_ = 0.1;
  double zeta_ = varpi_ * e_m_sq;
  double rho_ = varrho_ * varrho_;
  double omega_ = varpi_ * varpi_;
  this->initialize(sm.get_n_user(), sm.get_n_item(), n_component_,
                   sm.get_sparsity(), eta_, rho_, varrho_, zeta_, omega_,
                   varpi_, xi_, tau_, n_trunc_, sm.get_max_response() * 2);
}

void HPFBase::serialize(string& fname) {
  H5::H5File file(fname, H5F_ACC_TRUNC);
  vector<unsigned long int> uiparams(3);
  uiparams[0] = n_user;
  uiparams[1] = n_item;
  uiparams[2] = t_hyper;
  string dsetname = "uiparams";
  write_h5vector(file, dsetname, uiparams);

  vector<int> iparams(3);
  iparams[0] = n_component;
  iparams[1] = n_trunc;
  iparams[2] = val_max;
  dsetname = "iparams";
  write_h5vector(file, dsetname, iparams);

  vector<double> params(15);
  params[0] = eta;
  params[1] = eta_0;
  params[2] = rho;
  params[3] = rho_0;
  params[4] = varrho;
  params[5] = varrho_0;
  params[6] = zeta;
  params[7] = zeta_0;
  params[8] = omega;
  params[9] = omega_0;
  params[10] = varpi;
  params[11] = varpi_0;
  params[12] = xi;
  params[13] = tau;
  params[14] = sparsity;
  dsetname = "params";
  write_h5vector(file, dsetname, params);
  dsetname = "a_r";
  write_h5vector(file, dsetname, a_r);
  dsetname = "b_r";
  write_h5vector(file, dsetname, b_r);
  dsetname = "a_s";
  write_h5array2d(file, dsetname, a_s);
  dsetname = "b_s";
  write_h5array2d(file, dsetname, b_s);
  dsetname = "t_user";
  write_h5vector(file, dsetname, t_user);
  dsetname = "a_w";
  write_h5vector(file, dsetname, a_w);
  dsetname = "b_w";
  write_h5vector(file, dsetname, b_w);
  dsetname = "a_v";
  write_h5array2d(file, dsetname, a_v);
  dsetname = "b_v";
  write_h5array2d(file, dsetname, b_v);
  dsetname = "t_item";
  write_h5vector(file, dsetname, t_item);
  file.close();
}

void HPFBase::deserialize(string& fname) {
  H5::H5File file(fname, H5F_ACC_RDONLY);
  vector<unsigned long int> uiparams(3);
  string dsetname = "uiparams";
  read_h5vector(file, dsetname, uiparams);
  unsigned long int n_user_ = uiparams[0];
  unsigned long int n_item_ = uiparams[1];
  unsigned long int t_hyper_ = uiparams[1];

  vector<int> iparams(3);
  dsetname = "iparams";
  read_h5vector(file, dsetname, iparams);
  int n_component_ = iparams[0];
  int n_trunc_ = iparams[1];
  int val_max_ = iparams[2];

  vector<double> params(15);
  dsetname = "params";
  read_h5vector(file, dsetname, params);
  double eta_ = params[0];
  double eta_0_ = params[1];
  double rho_ = params[2];
  double rho_0_ = params[3];
  double varrho_ = params[4];
  double varrho_0_ = params[5];
  double zeta_ = params[6];
  double zeta_0_ = params[7];
  double omega_ = params[8];
  double omega_0_ = params[9];
  double varpi_ = params[10];
  double varpi_0_ = params[11];
  double xi_ = params[12];
  double tau_ = params[13];
  double sparsity_ = params[14];
  this->initialize(n_user_, n_item_, n_component_, sparsity_, eta_, rho_,
                   varrho_, zeta_, omega_, varpi_, xi_, tau_, n_trunc_,
                   val_max_);
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
  varrho_0 = varrho_0_;
  zeta_0 = zeta_0_;
  omega_0 = omega_0_;
  varpi_0 = varpi_0_;
  t_hyper = t_hyper_;
  file.close();
}

double HPFBase::get_eta() { return eta; }

double HPFBase::get_eta_0() { return eta_0; }

double HPFBase::get_rho() { return rho; }

double HPFBase::get_rho_0() { return rho_0; }

double HPFBase::get_varrho() { return varrho; }

double HPFBase::get_varrho_0() { return varrho_0; }

double HPFBase::get_zeta() { return zeta; }

double HPFBase::get_zeta_0() { return zeta_0; }

double HPFBase::get_omega() { return omega; }

double HPFBase::get_omega_0() { return omega_0; }

double HPFBase::get_varpi() { return varpi; }

double HPFBase::get_varpi_0() { return varpi_0; }

int HPFBase::get_n_trunc() { return n_trunc; }

int HPFBase::get_n_component() { return n_component; }

void HPFBase::set_eta(double eta_) { eta = eta_; }

void HPFBase::set_eta_0(double eta_0_) { eta_0 = eta_0_; }

void HPFBase::set_rho(double rho_) { rho = rho_; }

void HPFBase::set_rho_0(double rho_0_) { rho_0 = rho_0_; }

void HPFBase::set_varrho(double varrho_) { varrho = varrho_; }

void HPFBase::set_varrho_0(double varrho_0_) { varrho_0 = varrho_0_; }

void HPFBase::set_zeta(double zeta_) { zeta = zeta_; }

void HPFBase::set_zeta_0(double zeta_0_) { zeta_0 = zeta_0_; }

void HPFBase::set_omega(double omega_) { omega = omega_; }

void HPFBase::set_omega_0(double omega_0_) { omega_0 = omega_0_; }

void HPFBase::set_varpi(double varpi_) { varpi = varpi_; }

void HPFBase::set_varpi_0(double varpi_0_) { varpi_0 = varpi_0_; }

void HPFBase::set_n_trunc(int n_trunc_) { n_trunc = n_trunc_; }

double HPFBase::calc_log_lambda(unsigned long int ui, unsigned long int ii,
                                vector<double>& buffer_k) {
  if (n_component != buffer_k.size())
    cerr << "n_component=" << n_component << " buffer_k.size()=" << buffer_k.size() << endl;
  for (int j = 0; j < n_component; ++j)
    buffer_k[j] = log(a_s[ui][j]) - log(b_s[ui][j]) + log(a_v[ii][j]) - log(b_v[ii][j]);
  return logsumexp(buffer_k, n_component);
}

double HPFBase::calc_lru(unsigned long int ui) {
  t_user[ui] += 1;
  return pow(t_user[ui], -xi);
}

double HPFBase::calc_lri(unsigned long int ii) {
  t_item[ii] += 1;
  return pow(t_item[ii], -xi);
}

void HPFBase::calc_varphi(unsigned long int ui, unsigned long int ii,
                          vector<double>& varphi) {
  double norm = 0.0;
  for (int j = 0; j < n_component; ++j)
    varphi[j] = bmath::digamma(a_s[ui][j]) - log(b_s[ui][j]) +
                bmath::digamma(a_v[ii][j]) - log(b_v[ii][j]);
  norm = logsumexp(varphi, n_component);
  for (int j = 0; j < n_component; ++j)
    varphi[j] = exp(varphi[j] - norm);
}

void HPFBase::update_a_r(unsigned long int ui, double lru) {
  double gr = n_component * eta;
  a_r[ui] = (1 - lru) * a_r[ui] + lru * gr;
}

void HPFBase::update_b_r(unsigned long int ui, double lru) {
  double gr = 0.0;
  gr = rho / varrho;
  for (int j = 0; j < n_component; ++j) gr += a_s[ui][j] / b_s[ui][j];
  b_r[ui] = (1 - lru) * b_r[ui] + lru * gr;
}

void HPFBase::update_a_s(unsigned long int ui, double lru, double e_n,
                         vector<double>& varphi) {
  double gr = 0.0;
  for (int j = 0; j < n_component; ++j) {
    gr = eta + n_item * e_n * varphi[j];
    a_s[ui][j] = (1 - lru) * a_s[ui][j] + lru * gr;
  }
}

void HPFBase::update_a_s_with_zero(unsigned long int ui, double lru) {
  for (int j = 0; j < n_component; ++j)
    a_s[ui][j] = (1 - lru) * a_s[ui][j] + lru * eta;
}

void HPFBase::update_b_s(unsigned long int ui, unsigned long int ii, double lru,
                         double phi) {
  double gr = 0.0;
  for (int j = 0; j < n_component; ++j) {
    gr = a_r[ui] / b_r[ui] + n_item * phi * a_v[ii][j] / b_v[ii][j];
    b_s[ui][j] = (1 - lru) * b_s[ui][j] + lru * gr;
  }
}

void HPFBase::update_a_w(unsigned long int ii, double lri) {
  double gr = n_component * zeta;
  a_w[ii] = (1 - lri) * a_w[ii] + lri * gr;
}

void HPFBase::update_b_w(unsigned long int ii, double lri) {
  double gr = omega / varpi;
  for (int j = 0; j < n_component; ++j) gr += a_v[ii][j] / b_v[ii][j];
  b_w[ii] = (1 - lri) * b_w[ii] + lri * gr;
}

void HPFBase::update_a_v(unsigned long int ii, double lri, double e_n,
                         vector<double>& varphi) {
  double gr = 0.0;
  for (int j = 0; j < n_component; ++j) {
    gr = zeta + n_user * e_n * varphi[j];
    a_v[ii][j] = (1 - lri) * a_v[ii][j] + lri * gr;
  }
}

void HPFBase::update_a_v_with_zero(unsigned long int ii, double lri) {
  for (int j = 0; j < n_component; ++j)
    a_v[ii][j] = (1 - lri) * a_v[ii][j] + lri * zeta;
}

void HPFBase::update_b_v(unsigned long int ui, unsigned long int ii, double lri,
                         double phi) {
  double gr = 0.0;
  for (int j = 0; j < n_component; ++j) {
    gr = a_w[ii] / b_w[ii] + n_user * phi * a_s[ui][j] / b_s[ui][j];
    b_v[ii][j] = (1 - lri) * b_v[ii][j] + lri * gr;
  }
}

HyperParamStats* HPFBase::create_hyperparam_stats() {
  NullHyperParamStats* hps = new NullHyperParamStats();
  return hps;
}

void HPFBase::update_hyperparam(HyperParamStats* stats) {}

double HPFBase::predict(unsigned long int ui, unsigned long int ii,
                        vector<double>& buffer_k) {
  return exp(calc_log_lambda(ui, ii, buffer_k));
}

void HPFBase::update(unsigned long int ui, unsigned long int ii, double val,
                     vector<double>& buffer_k) {
  double lru = calc_lru(ui);
  double lri = calc_lri(ii);
  if (val == 0) {
    update_a_s_with_zero(ui, lru);
    update_b_s(ui, ii, lru, 1.0);
    update_a_r(ui, lru);
    update_b_r(ui, lru);
    update_a_v_with_zero(ii, lri);
    update_b_v(ui, ii, lri, 1.0);
    update_a_w(ii, lri);
    update_b_w(ii, lri);
  } else {
    double e_n = val;
    calc_varphi(ui, ii, buffer_k);
    update_a_s(ui, lru, e_n, buffer_k);
    update_b_s(ui, ii, lru, 1.0);
    update_a_r(ui, lru);
    update_b_r(ui, lru);
    update_a_v(ii, lri, e_n, buffer_k);
    update_b_v(ui, ii, lri, 1.0);
    update_a_w(ii, lri);
    update_b_w(ii, lri);
  }
}

void HPFBase::update(unsigned long int ui, unsigned long int ii, double val,
                     vector<double>& buffer_k, vector<double>& buffer_n,
                     HyperParamStats* stats) {
  update(ui, ii, val, buffer_k);
}

void HPFBase::test(unsigned long int ui, unsigned long int ii, double val,
                   vector<double>& buffer_k, test_result& res) {
  double log_lambda = calc_log_lambda(ui, ii, buffer_k);
  double m = exp(log_lambda);
  double cond_norm = bmath::expm1(m);
  double pred = m;
  double cond_pred = m * exp(m) / cond_norm;
  double tll = 0.0;
  if (val != 0) {
    tll = val * log_lambda - cache_log_fact[val];
  }
  res.pred = pred;
  res.test_loglik = tll - m;
  res.test_error = pow(val - pred, 2);
  res.cond_pred = cond_pred;
  res.cond_test_loglik = tll - log(cond_norm);
  res.cond_test_error = pow(val - cond_pred, 2);
}

void HPFBase::test(unsigned long int ui, unsigned long int ii, double val,
                   vector<double>& buffer_k, vector<double>& buffer_n,
                   test_result& res) {
  test(ui, ii, val, buffer_k, res);
}

HPF::HPF() {}

void HPF::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                        int n_trunc_, double xi_, double tau_) {
  HPFBase::initialize_with_sparse_matrix(sm, n_component_, n_trunc_, xi_, tau_);
}

void HPF::serialize(string& fname) { HPFBase::serialize(fname); }

void HPF::deserialize(string& fname) { HPFBase::deserialize(fname); }

int HPF::get_n_trunc() { return HPFBase::get_n_trunc(); }

int HPF::get_n_component() { return HPFBase::get_n_component(); }

HyperParamStats* HPF::create_hyperparam_stats() {
  return HPFBase::create_hyperparam_stats();
}

void HPF::update_hyperparam(HyperParamStats* stats) {
  HPFBase::update_hyperparam(stats);
}

double HPF::predict(unsigned long int ui, unsigned long int ii,
                    vector<double>& buffer_k) {
  return HPFBase::predict(ui, ii, buffer_k);
}

void HPF::update(unsigned long int ui, unsigned long int ii, double val,
                 vector<double>& buffer_k, vector<double>& buffer_n,
                 HyperParamStats* stats) {
  HPFBase::update(ui, ii, val, buffer_k, buffer_n, stats);
}

void HPF::test(unsigned long int ui, unsigned long int ii, double val,
               vector<double>& buffer_k, vector<double>& buffer_n,
               test_result& res) {
  HPFBase::test(ui, ii, val, buffer_k, buffer_n, res);
}

void HPF::update_q_n(unsigned long int ui, unsigned long int ii,
                     double log_lambda, int val, vector<double>& phi,
                     vector<double>& buffer_k, vector<double>& q_n) {
  double lambda = predict(ui, ii, buffer_k);
  for (int n = 1; n < n_trunc; ++n)
    q_n[n - 1] = val * log(lambda) + val * log(phi[n]) - lambda * phi[n] -
                 cache_log_fact[val] + n * log_lambda - cache_log_fact[n];
  double norm = logsumexp(q_n, n_trunc - 1);
  for (int n = n_trunc - 1; n !=0; --n) q_n[n] = exp(q_n[n-1] - norm);
  q_n[0] = 0.0;
}

void HPF::update_q_n(unsigned long int ui, unsigned long int ii,
                     double log_lambda, double val, vector<double>& phi,
                     vector<double>& buffer_k, vector<double>& q_n) {
  update_q_n(ui, ii, log_lambda, (int)val, phi, buffer_k, q_n);
}

void HPF::update(unsigned long int ui, unsigned long int ii, double val,
                 vector<double>& buffer_k, double e_phi) {
  double lru = calc_lru(ui);
  double lri = calc_lri(ii);
  if (val == 0) {
    update_a_s_with_zero(ui, lru);
    update_b_s(ui, ii, lru, 1.0);
    update_a_r(ui, lru);
    update_b_r(ui, lru);
    update_a_v_with_zero(ii, lri);
    update_b_v(ui, ii, lri, 1.0);
    update_a_w(ii, lri);
    update_b_w(ii, lri);
  } else {
    double e_n = val;
    calc_varphi(ui, ii, buffer_k);
    update_a_s(ui, lru, e_n, buffer_k);
    update_b_s(ui, ii, lru, e_phi);
    update_a_r(ui, lru);
    update_b_r(ui, lru);
    update_a_v(ii, lri, e_n, buffer_k);
    update_b_v(ui, ii, lri, e_phi);
    update_a_w(ii, lri);
    update_b_w(ii, lri);
  }
}

void HPF::update(unsigned long int ui, unsigned long int ii, double val,
                 vector<double>& buffer_k, vector<double>& buffer_n,
                 HyperParamStats* stats, double e_phi) {
  update(ui, ii, val, buffer_k, e_phi);
}

void HPF::update_test_result(unsigned long int ui, unsigned long int ii,
                             double log_lambda, int val,
                             vector<double>& buffer_k, vector<double>& buffer_n,
                             vector<double>& phi, test_result& res) {
  double lambda = this->predict(ui, ii, buffer_k);
  double element_mean = lambda;
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

void HPF::update_test_result(unsigned long int ui, unsigned long int ii,
                             double log_lambda, double val,
                             vector<double>& buffer_k, vector<double>& buffer_n,
                             vector<double>& phi, test_result& res) {
  update_test_result(ui, ii, log_lambda, (int)val, buffer_k, buffer_n, phi,
                     res);
}
