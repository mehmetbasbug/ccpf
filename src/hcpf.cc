#include "utils.h"
#include "hcpf.h"
using namespace std;
namespace bmath = boost::math;

const double CUTOFF_LOGLIK = log(1e-6);
const int N_TRUNC_MAX = 100;

HCPF::HCPF() {}

HCPF::HCPF(unsigned long int n_user_, unsigned long int n_item_,
           int n_component_, double sparsity_, double eta_, double rho_,
           double varrho_, double zeta_, double omega_, double varpi_,
           double xi_, double tau_, int n_trunc_, int val_max_) {
  initialize(n_user_, n_item_, n_component_, sparsity_, eta_, rho_, varrho_,
             zeta_, omega_, varpi_, xi_, tau_, n_trunc_, val_max_);
}

void HCPF::initialize(unsigned long int n_user_, unsigned long int n_item_,
                      int n_component_, double sparsity_, double eta_,
                      double rho_, double varrho_, double zeta_, double omega_,
                      double varpi_, double xi_, double tau_, int n_trunc_,
                      int val_max_) {
  HPFBase::initialize(n_user_, n_item_, n_component_, sparsity_, eta_, rho_,
                      varrho_, zeta_, omega_, varpi_, xi_, tau_, n_trunc_,
                      val_max_);
  phi.resize(n_trunc_);
  for (int n = 0; n < n_trunc_; ++n)
    phi[n] = n;
}

void HCPF::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                         double xi_, double tau_) {
  double e_m = -log(sm.get_sparsity());
  int n_trunc_ = N_TRUNC_MAX;
  double cur_loglik;
  do {
    n_trunc_ -= 1;
    cur_loglik = n_trunc_ * log(e_m) - log(bmath::expm1(e_m)) -
                 bmath::lgamma(n_trunc_ + 1);
  } while (CUTOFF_LOGLIK > cur_loglik);
  n_trunc_ += 1;
  double e_m_sq = sqrt(e_m / (double)n_component_);
  double varrho_ = 0.1;
  double eta_ = varrho_ * e_m_sq;
  double varpi_ = 0.1;
  double zeta_ = varpi_ * e_m_sq;
  double rho_ = varrho_ * varrho_;
  double omega_ = varpi_ * varpi_;
  initialize(sm.get_n_user(), sm.get_n_item(), n_component_,
                      sm.get_sparsity(), eta_, rho_, varrho_, zeta_, omega_,
                      varpi_, xi_, tau_, n_trunc_, sm.get_max_response() * 2);
}

void HCPF::serialize(string& fname) { HPFBase::serialize(fname); }

void HCPF::deserialize(string& fname) {
  HPFBase::deserialize(fname);
  phi.resize(n_trunc);
  for (int n = 0; n < n_trunc; ++n)
    phi[n] = n;
}

int HCPF::get_n_trunc() { return n_trunc; }

int HCPF::get_n_component() {return n_component; }

vector<double>& HCPF::get_phi() { return phi; }

double HCPF::calc_log_lambda(unsigned long int ui, unsigned long int ii,
                             vector<double>& buffer_k) {
  HPFBase::calc_log_lambda(ui, ii, buffer_k);
}

void HCPF::calc_scaling(vector<double>& q_n, double& e_n, double& e_phi) {
  e_n = 0.0;
  for (int n = 1; n < n_trunc; ++n) e_n += n * q_n[n];
  e_phi = e_n;
}

double HCPF::predict(unsigned long int ui, unsigned long int ii,
                     vector<double>& buffer_k) {
  HPFBase::predict(ui, ii, buffer_k);
}

HyperParamStats* HCPF::create_hyperparam_stats() {
  NullHyperParamStats* hps = new NullHyperParamStats();
  return hps;
}

void HCPF::update_hyperparam(HyperParamStats* stats) {}

void HCPF::update(unsigned long int ui, unsigned long int ii, double val,
                  vector<double>& buffer_k, double e_n) {
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

void HCPF::update(unsigned long int ui, unsigned long int ii, double val,
                  vector<double>& buffer_k, double e_n, double e_phi,
                  double e_mu, HyperParamStats* stats) {
  update(ui, ii, val, buffer_k, e_n);
}

HCPFNormal::HCPFNormal() {}

HCPFNormal::HCPFNormal(unsigned long int n_user_, unsigned long int n_item_,
                         int n_component_, double sparsity_, double eta_,
                         double rho_, double varrho_, double zeta_,
                         double omega_, double varpi_, double xi_, double tau_,
                         int n_trunc_, int val_max_) {
  initialize(n_user_, n_item_, n_component_, sparsity_, eta_, rho_, varrho_,
             zeta_, omega_, varpi_, xi_, tau_, n_trunc_, val_max_);
}

void HCPFNormal::initialize(unsigned long int n_user_,
                             unsigned long int n_item_, int n_component_,
                             double sparsity_, double eta_, double rho_,
                             double varrho_, double zeta_, double omega_,
                             double varpi_, double xi_, double tau_,
                             int n_trunc_, int val_max_) {
  HPFBase::initialize(n_user_, n_item_, n_component_, sparsity_, eta_, rho_,
                      varrho_, zeta_, omega_, varpi_, xi_, tau_, n_trunc_,
                      val_max_);
  phi_mean.resize(n_trunc);
  phi_var.resize(n_trunc);
  for (int n = 0; n < n_trunc; ++n) {
    phi_mean[n] = n;
    phi_var[n] = n;
  }
}

void HCPFNormal::initialize_with_sparse_matrix(SparseMatrix& sm,
                                                int n_component_, double xi_,
                                                double tau_) {
  double e_m = -log(sm.get_sparsity());
  int n_trunc_ = N_TRUNC_MAX;
  double cur_loglik;
  do {
    n_trunc_ -= 1;
    cur_loglik = n_trunc_ * log(e_m) - log(bmath::expm1(e_m)) -
                 bmath::lgamma(n_trunc_ + 1);
  } while (CUTOFF_LOGLIK > cur_loglik);
  n_trunc_ += 1;
  double e_m_sq = sqrt(e_m / (double)n_component_);
  double varrho_ = 0.1;
  double eta_ = varrho_ * e_m_sq;
  double varpi_ = 0.1;
  double zeta_ = varpi_ * e_m_sq;
  double rho_ = varrho_ * varrho_;
  double omega_ = varpi_ * varpi_;
  initialize(sm.get_n_user(), sm.get_n_item(), n_component_,
             sm.get_sparsity(), eta_, rho_, varrho_, zeta_, omega_,
             varpi_, xi_, tau_, n_trunc_, sm.get_max_response() * 2);
}

void HCPFNormal::serialize(string& fname) { HPFBase::serialize(fname); }

void HCPFNormal::deserialize(string& fname) {
  HPFBase::deserialize(fname);
  phi_mean.resize(n_trunc);
  phi_var.resize(n_trunc);
  for (int n = 0; n < n_trunc; ++n) {
    phi_mean[n] = n;
    phi_var[n] = n;
  }
}

int HCPFNormal::get_n_trunc() { return n_trunc; }

int HCPFNormal::get_n_component() {return n_component; }

vector<double>& HCPFNormal::get_phi_mean() { return phi_mean; }

vector<double>& HCPFNormal::get_phi_var() { return phi_var; }

double HCPFNormal::calc_log_lambda(unsigned long int ui, unsigned long int ii,
                                    vector<double>& buffer_k) {
  HPFBase::calc_log_lambda(ui, ii, buffer_k);
}

void HCPFNormal::calc_scaling(vector<double>& q_n, double& e_n,
                               double& e_phi_mean, double& e_phi_var) {
  e_n = 0.0;
  for (int n = 1; n < n_trunc; ++n) e_n += n * q_n[n];
  e_phi_mean = e_n;
  e_phi_var = e_n;
}

double HCPFNormal::predict(unsigned long int ui, unsigned long int ii,
                            vector<double>& buffer_k) {
  HPFBase::predict(ui, ii, buffer_k);
}

HyperParamStats* HCPFNormal::create_hyperparam_stats() {
  NullHyperParamStats* hps = new NullHyperParamStats();
  return hps;
}

void HCPFNormal::update_hyperparam(HyperParamStats* stats) {}

void HCPFNormal::update(unsigned long int ui, unsigned long int ii, double val,
                         vector<double>& buffer_k, double e_n) {
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

void HCPFNormal::update(unsigned long int ui, unsigned long int ii, double val,
                         vector<double>& buffer_k, double e_n,
                         double e_phi_mean, double e_phi_var, double e_mu,
                         double e_sigma2, HyperParamStats* stats) {
  update(ui, ii, val, buffer_k, e_n);
}

HICPFHyperParamStats::HICPFHyperParamStats() {}

void HICPFHyperParamStats::initialize() {
  CM_cov = 0;
  CM_var = 100000;
}

void HICPFHyperParamStats::update_cm_cov(double x) { CM_cov += x; }

void HICPFHyperParamStats::update_cm_var(double x) { CM_var += x; }

double HICPFHyperParamStats::get_cm_gradient() {
  return CM_cov / CM_var;
}

HICPF::HICPF() {}

HICPF::HICPF(unsigned long int n_user_, unsigned long int n_item_,
             int n_component_, double sparsity_, double eta_,
             double rho_, double varrho_, double zeta_,
             double omega_, double varpi_, double xi_,
             double tau_, int n_trunc_, int val_max_) {
  initialize(n_user_, n_item_, n_component_, sparsity_, eta_, rho_, varrho_,
             zeta_, omega_, varpi_, xi_, tau_, n_trunc_, val_max_);
}

void HICPF::initialize(unsigned long int n_user_,
                       unsigned long int n_item_, int n_component_,
                       double sparsity_, double eta_, double rho_,
                       double varrho_, double zeta_, double omega_,
                       double varpi_, double xi_, double tau_,
                       int n_trunc_, int val_max_) {
  HPFBase::initialize(n_user_, n_item_, n_component_, sparsity_, eta_, rho_,
                      varrho_, zeta_, omega_, varpi_, xi_, tau_, n_trunc_,
                      val_max_);
  phi.resize(n_trunc_);
  for (int n = 0; n < n_trunc_; ++n)
    phi[n] = 1;
  cm = 0;
}

void HICPF::initialize_with_sparse_matrix(SparseMatrix& sm,
                                                 int n_component_, double xi_,
                                                 double tau_) {
  double e_m = -log(sm.get_sparsity());
  int n_trunc_ = N_TRUNC_MAX;
  double cur_loglik;
  do {
    n_trunc_ -= 1;
    cur_loglik = n_trunc_ * log(e_m) - log(bmath::expm1(e_m)) -
                 bmath::lgamma(n_trunc_ + 1);
  } while (CUTOFF_LOGLIK > cur_loglik);
  n_trunc_ += 1;
  double e_m_sq = sqrt(e_m / (double)n_component_);
  double varrho_ = 0.1;
  double eta_ = varrho_ * e_m_sq;
  double varpi_ = 0.1;
  double zeta_ = varpi_ * e_m_sq;
  double rho_ = varrho_ * varrho_;
  double omega_ = varpi_ * varpi_;
  initialize(sm.get_n_user(), sm.get_n_item(), n_component_,
                      sm.get_sparsity(), eta_, rho_, varrho_, zeta_, omega_,
                      varpi_, xi_, tau_, n_trunc_, sm.get_max_response() * 2);
}

void HICPF::serialize(string& fname) {
  // TODO fix deserialize to save cm
  HPFBase::serialize(fname);
}

void HICPF::deserialize(string& fname) {
  // TODO fix deserialize to load cm
  HPFBase::deserialize(fname);
  phi.resize(n_trunc);
  for (int n = 0; n < n_trunc; ++n)
    phi[n] = 1;
  cm = 0;
}

int HICPF::get_n_trunc() { return n_trunc; }

int HICPF::get_n_component() {return n_component; }

vector<double>& HICPF::get_phi() { return phi; }

double HICPF::calc_log_lambda(unsigned long int ui, unsigned long int ii,
                                     vector<double>& buffer_k) {
  HPFBase::calc_log_lambda(ui, ii, buffer_k);
}

void HICPF::calc_scaling(vector<double>& q_n, double& e_n, double& e_phi) {
  e_n = 0.0;
  for (int n = 1; n < n_trunc; ++n) e_n += n * q_n[n];
  e_phi = 1 - cm + cm * e_n;
}

double HICPF::predict(unsigned long int ui, unsigned long int ii,
                             vector<double>& buffer_k) {
  HPFBase::predict(ui, ii, buffer_k);
}

HyperParamStats* HICPF::create_hyperparam_stats() {
  HICPFHyperParamStats* hps = new HICPFHyperParamStats();
  hps->initialize();
  return hps;
}

void HICPF::update_hyperparam(HyperParamStats* stats) {
  update_hyperparam(dynamic_cast<HICPFHyperParamStats*>(stats));
}

void HICPF::update_hyperparam(HICPFHyperParamStats* stats) {
  t_hyper += 1;
  double lr = pow(t_hyper, -xi);
  cm = (1 - lr) * cm + lr * stats->get_cm_gradient();
  stats->initialize();
}

void HICPF::update(unsigned long int ui, unsigned long int ii,
                          double val, vector<double>& buffer_k, double e_n,
                          double e_phi, double e_mu,
                          HICPFHyperParamStats* stats) {
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
    calc_varphi(ui, ii, buffer_k);
    update_a_s(ui, lru, e_n, buffer_k);
    update_b_s(ui, ii, lru, 1.0);
    update_a_r(ui, lru);
    update_b_r(ui, lru);
    update_a_v(ii, lri, e_n, buffer_k);
    update_b_v(ui, ii, lri, 1.0);
    update_a_w(ii, lri);
    update_b_w(ii, lri);

    double sq_err = pow(val - e_mu * e_phi, 2);
    double sq_reg = pow(e_n - 1, 2);
    stats->update_cm_cov((val - e_mu) * (e_n - 1) / e_mu);
    stats->update_cm_var(sq_reg);
  }
}

void HICPF::update(unsigned long int ui, unsigned long int ii, double val,
                   vector<double>& buffer_k, double e_n, double e_phi,
                   double e_mu, HyperParamStats* stats) {
  update(ui, ii, val, buffer_k, e_n, e_phi, e_mu,
         dynamic_cast<HICPFHyperParamStats*>(stats));
}

HICPFNormalHyperParamStats::HICPFNormalHyperParamStats() {}

void HICPFNormalHyperParamStats::initialize() {
  CM_cov = 0;
  CM_var = 100000;
  CV_cov = 0;
  CV_var = 100000;
}

void HICPFNormalHyperParamStats::update_cm_cov(double x) { CM_cov += x; }

void HICPFNormalHyperParamStats::update_cm_var(double x) { CM_var += x; }

void HICPFNormalHyperParamStats::update_cv_cov(double x) { CV_cov += x; }

void HICPFNormalHyperParamStats::update_cv_var(double x) { CV_var += x; }

double HICPFNormalHyperParamStats::get_cm_gradient() { return CM_cov / CM_var; }

double HICPFNormalHyperParamStats::get_cv_gradient() { return CV_cov / CV_var; }

HICPFNormal::HICPFNormal() {}

HICPFNormal::HICPFNormal(unsigned long int n_user_, unsigned long int n_item_,
                         int n_component_, double sparsity_, double eta_,
                         double rho_, double varrho_, double zeta_,
                         double omega_, double varpi_, double xi_, double tau_,
                         int n_trunc_, int val_max_) {
  initialize(n_user_, n_item_, n_component_, sparsity_, eta_, rho_, varrho_,
             zeta_, omega_, varpi_, xi_, tau_, n_trunc_, val_max_);
}

void HICPFNormal::initialize(unsigned long int n_user_,
                             unsigned long int n_item_, int n_component_,
                             double sparsity_, double eta_, double rho_,
                             double varrho_, double zeta_, double omega_,
                             double varpi_, double xi_, double tau_,
                             int n_trunc_, int val_max_) {
  HPFBase::initialize(n_user_, n_item_, n_component_, sparsity_, eta_, rho_,
                      varrho_, zeta_, omega_, varpi_, xi_, tau_, n_trunc_,
                      val_max_);
  phi_mean.resize(n_trunc_);
  phi_var.resize(n_trunc_);
  for (int n = 0; n < n_trunc_; ++n) {
    phi_mean[n] = 1;
    phi_var[n] = 1;
  }
  cm = 0;
  cv = 0;
}

void HICPFNormal::initialize_with_sparse_matrix(SparseMatrix& sm,
                                                int n_component_, double xi_,
                                                double tau_) {
  double e_m = -log(sm.get_sparsity());
  int n_trunc_ = N_TRUNC_MAX;
  double cur_loglik;
  do {
    n_trunc_ -= 1;
    cur_loglik = n_trunc_ * log(e_m) - log(bmath::expm1(e_m)) -
                 bmath::lgamma(n_trunc_ + 1);
  } while (CUTOFF_LOGLIK > cur_loglik);
  n_trunc_ += 1;
  double e_m_sq = sqrt(e_m / (double)n_component_);
  double varrho_ = 0.1;
  double eta_ = varrho_ * e_m_sq;
  double varpi_ = 0.1;
  double zeta_ = varpi_ * e_m_sq;
  double rho_ = varrho_ * varrho_;
  double omega_ = varpi_ * varpi_;
  initialize(sm.get_n_user(), sm.get_n_item(), n_component_,
                      sm.get_sparsity(), eta_, rho_, varrho_, zeta_, omega_,
                      varpi_, xi_, tau_, n_trunc_, sm.get_max_response() * 2);
}

void HICPFNormal::serialize(string& fname) { HPFBase::serialize(fname); }

void HICPFNormal::deserialize(string& fname) {
  HPFBase::deserialize(fname);
  phi_mean.resize(n_trunc);
  phi_var.resize(n_trunc);
  for (int n = 0; n < n_trunc; ++n) {
    phi_mean[n] = 1 - cm + cm * n;
    phi_var[n] = 1 - cv + cv * n;
  }
  cm = 0;
  cv = 0;
}

int HICPFNormal::get_n_trunc() { return n_trunc; }

int HICPFNormal::get_n_component() {return n_component; }

vector<double>& HICPFNormal::get_phi_mean() { return phi_mean; }

vector<double>& HICPFNormal::get_phi_var() { return phi_var; }

double HICPFNormal::calc_log_lambda(unsigned long int ui, unsigned long int ii,
                                    vector<double>& buffer_k) {
  HPFBase::calc_log_lambda(ui, ii, buffer_k);
}

void HICPFNormal::calc_scaling(vector<double>& q_n, double& e_n,
                               double& e_phi_mean, double& e_phi_var) {
  e_n = 0.0;
  for (int n = 1; n < n_trunc; ++n) e_n += n * q_n[n];
  e_phi_mean = 1 - cm + cm * e_n;
  e_phi_var = 1 - cv + cv * e_n;
}

double HICPFNormal::predict(unsigned long int ui, unsigned long int ii,
                            vector<double>& buffer_k) {
  HPFBase::predict(ui, ii, buffer_k);
}

HyperParamStats* HICPFNormal::create_hyperparam_stats() {
  HICPFNormalHyperParamStats* hps = new HICPFNormalHyperParamStats();
  hps->initialize();
  return hps;
}

void HICPFNormal::update_hyperparam(HICPFNormalHyperParamStats* stats) {
  t_hyper += 1;
  double lr = pow(t_hyper, -xi);
  cm = (1 - lr) * cm + lr * stats->get_cm_gradient();
  cv = (1 - lr) * cv + lr * stats->get_cv_gradient();
  stats->initialize();
}

void HICPFNormal::update_hyperparam(HyperParamStats* stats) {
  update_hyperparam(dynamic_cast<HICPFNormalHyperParamStats*>(stats));
}

void HICPFNormal::update(unsigned long int ui, unsigned long int ii, double val,
                         vector<double>& buffer_k, double e_n,
                         double e_phi_mean, double e_phi_var, double e_mu,
                         double e_sigma2, HICPFNormalHyperParamStats* stats) {
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
    calc_varphi(ui, ii, buffer_k);
    update_a_s(ui, lru, e_n, buffer_k);
    update_b_s(ui, ii, lru, 1.0);
    update_a_r(ui, lru);
    update_b_r(ui, lru);
    update_a_v(ii, lri, e_n, buffer_k);
    update_b_v(ui, ii, lri, 1.0);
    update_a_w(ii, lri);
    update_b_w(ii, lri);

    double sq_err = pow(val - e_mu * e_phi_mean, 2);
    double sq_reg = pow(e_n - 1, 2);
    stats->update_cm_cov((val - e_mu) * (e_n - 1) / e_mu);
    stats->update_cm_var(sq_reg);
    stats->update_cv_cov((sq_err - e_sigma2) * (e_n - 1) / e_sigma2);
    stats->update_cv_var(sq_reg);
  }
}

void HICPFNormal::update(unsigned long int ui, unsigned long int ii, double val,
                         vector<double>& buffer_k, double e_n,
                         double e_phi_mean, double e_phi_var, double e_mu,
                         double e_sigma2, HyperParamStats* stats) {
  update(ui, ii, val, buffer_k, e_n, e_phi_mean, e_phi_var, e_mu, e_sigma2,
         dynamic_cast<HICPFNormalHyperParamStats*>(stats));
}

// HCPFBase::HCPFBase() {}

// void HCPFBase::initialize_with_sparse_matrix(SparseMatrix& sm,
//                                              int n_component_,
//                                              int n_trunc_,
//                                              double xi_ = 0.7,
//                                              double tau_ = 10000) {
//   double e_m_sq = sqrt(-log(sm.get_sparsity()) / (double)n_component_);
//   double varrho_ = 0.1;
//   double eta_ = varrho_ * e_m_sq;
//   double varpi_ = 0.1;
//   double zeta_ = varpi_ * e_m_sq;
//   double rho_ = varrho_ * varrho_;
//   double omega_ = varpi_ * varpi_;
//   this->initialize(sm.get_n_user(), sm.get_n_item(), n_component_,
//                    sm.get_sparsity(), eta_, rho_, varrho_, zeta_, omega_,
//                    varpi_, xi_, tau_, n_trunc_, sm.get_max_response() * 2);
// }

// void HCPFBase::update(unsigned long int ui, unsigned long int ii, double val,
//                       vector<double>& buffer_k, vector<double>& q_n) {
//   double lru = calc_lru(ui);
//   double lri = calc_lri(ii);
//   if (val == 0) {
//     update_a_s_with_zero(ui, lru);
//     update_b_s(ui, ii, lru, phi);
//     update_a_r(ui, lru);
//     update_b_r(ui, lru);
//     update_a_v_with_zero(ii, lri);
//     update_b_v(ui, ii, lri, phi);
//     update_a_w(ii, lri);
//     update_b_w(ii, lri);
//   } else {
//     double e_n, e_phi_mean, e_phi_var;
//     calc_scaling(q_n, e_n, e_phi_mean, e_phi_var);
//     calc_varphi(ui, ii, buffer_k);
//     update_a_s(ui, lru, e_n, buffer_k);
//     update_b_s(ui, ii, lru, phi);
//     update_a_r(ui, lru);
//     update_b_r(ui, lru);
//     update_a_v(ii, lri, e_n, buffer_k);
//     update_b_v(ui, ii, lri, phi);
//     update_a_w(ii, lri);
//     update_b_w(ii, lri);
//   }
// }

// vector<double>& HCPFBase::get_phi_mean() { return phi_mean; }

// vector<double>& HCPFBase::get_phi_var() { return phi_var; }

// HCPF::HCPF() {}

// void HCPF::calc_scaling(vector<double>& q_n, double& e_n, double& e_phi_mean,
// double& e_phi_var){
//     e_n = 0.0;
//     for (int n = 1; n < n_trunc; ++n)
//       e_n += n * q_n[n];
//     e_phi_mean = e_n;
//     e_phi_var = e_n;
// }

// void HCPF::update_hyperparams(double val,
//                               double e_phi_mean,
//                               double e_phi_var,
//                               HCPFHyperParamStats* stats){
//     stats->update_cm_cov((val - lambda_)*(e_n-1) / lambda_);
//     stats->update_cm_var(sq_reg);
// }

// CCPF::CCPF() {}

// void CCPF::calc_scaling(vector<double>& q_n, double& e_n, double& e_phi_mean,
// double& e_phi_var){
//     e_n = 0.0;
//     for (int n = 1; n < n_trunc; ++n)
//       e_n += n * q_n[n];
//     e_phi_mean = 1 - cm + cm*e_n;
//     e_phi_var = 1 - cv + cv*e_n;
// }

// void CCPF::update_hyperparams(double val,
//                               double e_phi_mean,
//                               double e_phi_var,
//                               HCPFHyperParamStats* stats){
//     stats->update_cm_cov((val - lambda_)*(e_n-1) / lambda_);
//     stats->update_cm_var(sq_reg);
// }

// CCPF::CCPF() {}
// CCPF::CCPF(unsigned long int n_user_, unsigned long int n_item_, int K_,
//            int n_trunc_, int val_max_, double sm_upsilon_, double sm_varrho_,
//            double sm_eta_, double sm_varpi_, double sm_zeta_, double sm_rho_,
//            double sm_omega_, double rm_upsilon_, double rm_varrho_,
//            double rm_eta_, double rm_varpi_, double rm_zeta_, double rm_rho_,
//            double rm_omega_, double xi_, double tau_)
//     : rmdl(n_user_, n_item_, K_, n_trunc_, val_max_, rm_upsilon_, rm_varrho_,
//            rm_eta_, rm_varpi_, rm_zeta_, rm_rho_, rm_omega_, xi_, tau_),
//       smdl(n_user_, n_item_, K_, n_trunc_, val_max_, sm_upsilon_, 1.0,
//            sm_varrho_, sm_eta_, sm_varpi_, sm_zeta_, sm_rho_, sm_omega_, xi_,
//            tau_) {}

// void CCPF::initialize(SparseMatrix& sm, double xi_, double tau_) {
//   this->smdl.initialize_with_sparse_matrix(sm, xi_, tau_);
//   int n_trunc_ =

//   this->rmdl.initialize_with_sparse_matrix(sm, n_trunc_, xi_, tau_);
// }

// void CCPF::serialize(string& fname) {
//   string rm_fname;
//   rm_fname = fname;
//   rm_fname.insert(rm_fname.length() - 3, "_rm");
//   this->rmdl.serialize(rm_fname);
//   string sm_fname;
//   sm_fname = fname;
//   sm_fname.insert(sm_fname.length() - 3, "_sm");
//   this->smdl.serialize(sm_fname);
// }

// void CCPF::deserialize(string& fname) {
//   string rm_fname;
//   rm_fname = fname;
//   rm_fname.insert(rm_fname.length() - 3, "_rm");
//   this->rmdl.deserialize(rm_fname);
//   string sm_fname;
//   sm_fname = fname;
//   sm_fname.insert(sm_fname.length() - 3, "_sm");
//   this->smdl.deserialize(sm_fname);
// }
