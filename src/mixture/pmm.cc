#include "utils.h"
#include "mixture/pmm.h"
using namespace std;
namespace bmath = boost::math;

PMMHyperParamStats::PMMHyperParamStats() {}
void PMMHyperParamStats::initialize(double rho_0) {
  // Gamma prior on rho
  rho_a = rho_0;
  rho_b = 1.0;
}
void PMMHyperParamStats::update_rho_a(double sum_alpha) { rho_a += sum_alpha; }
void PMMHyperParamStats::update_rho_b(double sum_x) { rho_b += sum_x; }
double PMMHyperParamStats::get_rho_gradient() { return rho_a / rho_b; }

PMM::PMM() {}
PMM::PMM(unsigned long int dim1_, unsigned long int dim2_, int n_component_,
         double eta_, double rho_, double xi_, double tau_, int n_trunc_,
         int val_max_) {
  this->initialize(dim1_, dim2_, n_component_, eta_, rho_, xi_, tau_, n_trunc_,
                   val_max_);
}

void PMM::initialize(unsigned long int dim1_, unsigned long int dim2_,
                     int n_component_, double eta_, double rho_, double xi_,
                     double tau_, int n_trunc_, int val_max_) {
  dim1 = dim1_;
  dim2 = dim2_;
  n_component = n_component_;
  eta = eta_;
  eta_0 = eta;
  rho = rho_;
  rho_0 = rho;
  n_trunc = n_trunc_;
  xi = xi_;
  tau = tau_;
  val_max = val_max_;
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
  int max_dim = max(val_max, n_trunc);
  cache_log_fact.resize(max_dim);
  for (int j = 0; j < max_dim; ++j) cache_log_fact[j] = bmath::lgamma(j + 1);
}

void PMM::serialize(string& fname) {
  H5::H5File file(fname, H5F_ACC_TRUNC);
  vector<unsigned long int> uiparams(3);
  uiparams[0] = dim1;
  uiparams[1] = dim2;
  uiparams[2] = t_hyper;
  string dsetname = "uiparams";
  write_h5vector(file, dsetname, uiparams);

  vector<int> iparams(3);
  iparams[0] = n_component;
  iparams[1] = n_trunc;
  iparams[2] = val_max;
  dsetname = "iparams";
  write_h5vector(file, dsetname, iparams);

  vector<double> params(6);
  params[0] = eta;
  params[1] = eta_0;
  params[2] = rho;
  params[3] = rho_0;
  params[4] = xi;
  params[5] = tau;
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

void PMM::deserialize(string& fname) {
  H5::H5File file(fname, H5F_ACC_RDONLY);
  vector<unsigned long int> uiparams(3);
  string dsetname = "uiparams";
  read_h5vector(file, dsetname, uiparams);
  unsigned long int dim1_ = uiparams[0];
  unsigned long int dim2_ = uiparams[1];
  unsigned long int t_hyper_ = uiparams[1];

  vector<int> iparams(3);
  dsetname = "iparams";
  read_h5vector(file, dsetname, iparams);
  int n_component_ = iparams[0];
  int n_trunc_ = iparams[1];
  int val_max_ = iparams[2];

  vector<double> params(6);
  dsetname = "params";
  read_h5vector(file, dsetname, params);
  double rho_ = params[0];
  double rho_0_ = params[1];
  double eta_ = params[2];
  double eta_0_ = params[3];
  double xi_ = params[4];
  double tau_ = params[5];
  this->initialize(dim1_, dim2_, n_component_, eta_, rho_, xi_, tau_, n_trunc_,
                   val_max_);
  dsetname = "a_s";
  read_h5array2d(file, dsetname, a_s);
  dsetname = "b_s";
  read_h5array2d(file, dsetname, b_s);
  dsetname = "t_learning";
  read_h5vector(file, dsetname, t_learning);
  eta_0 = eta_0_;
  rho_0 = rho_0_;
  t_hyper = t_hyper_;
  file.close();
}

double PMM::get_eta() { return eta; }

double PMM::get_eta_0() { return eta_0; }

double PMM::get_rho() { return rho; }

double PMM::get_rho_0() { return rho_0; }

int PMM::get_n_trunc() { return n_trunc; }

int PMM::get_n_component() { return n_component; }

void PMM::set_eta(double eta_) { eta = eta_; }

void PMM::set_eta_0(double eta_0_) { eta_0 = eta_0_; }

void PMM::set_rho(double rho_) { rho = rho_; }

void PMM::set_rho_0(double rho_0_) { rho_0 = rho_0_; }

double PMM::calc_lr(unsigned long int ind) {
  t_learning[ind] += 1;
  return pow(t_learning[ind], -xi);
}

void PMM::calc_varphi(unsigned long int ind, vector<double>& varphi) {
  double norm = 0.0;
  for (int j = 0; j < n_component; ++j)
    varphi[j] = bmath::digamma(a_s[ind][j]) - log(b_s[ind][j]);
  norm = logsumexp(varphi, n_component);
  for (int j = 0; j < n_component; ++j) varphi[j] = exp(varphi[j] - norm);
}

void PMM::update_a_s(unsigned long int ind, double lr, double val,
                     vector<double>& varphi) {
  for (int j = 0; j < n_component; ++j)
    a_s[ind][j] = (1 - lr) * a_s[ind][j] + lr * (eta + dim2 * val * varphi[j]);
}

void PMM::update_b_s(unsigned long int ind, double lr, double phi = 1.0) {
  for (int j = 0; j < n_component; ++j)
    b_s[ind][j] = (1 - lr) * b_s[ind][j] + lr * (tau + dim2 * phi);
}

HyperParamStats* PMM::create_hyperparam_stats() {
  PMMHyperParamStats* hps = new PMMHyperParamStats();
  hps->initialize(rho_0);
  return hps;
}

void PMM::update_hyperparam(PMMHyperParamStats* stats) {
  t_hyper += 1;
  double lr = pow(t_hyper, -xi);
  rho = (1 - lr) * rho + lr * stats->get_rho_gradient();
  stats->initialize(rho_0);
}

double PMM::predict(unsigned long int ind, vector<double>& buffer_k) {
  if (n_component != buffer_k.size())
    cout << "n_component="<<n_component<<" buffer_k.size()="<<buffer_k.size()<<endl;
  for (int j = 0; j < n_component; ++j)
    buffer_k[j] = log(a_s[ind][j]) - log(b_s[ind][j]);
  return exp(logsumexp(buffer_k, n_component));
}

void PMM::update(unsigned long int ind, int val, vector<double>& buffer_k,
                 PMMHyperParamStats* stats) {
  double lr = calc_lr(ind);
  calc_varphi(ind, buffer_k);
  update_a_s(ind, lr, val, buffer_k);
  update_b_s(ind, lr, 1.0);

  // TODO Review these updates
  double sum = 0.0;
  for (int j = 0; j < n_component; ++j) sum += b_s[ind][j];
  stats->update_rho_a(n_component * eta);
  stats->update_rho_b(sum);
}

void PMM::test(unsigned long int ind, int val, vector<double>& buffer_k,
               test_result& res) {
  double m = predict(ind, buffer_k);
  double cond_norm = bmath::expm1(m);
  double pred = m;
  double cond_pred = m * exp(m) / cond_norm;
  double tll = val * log(m) - cache_log_fact[val];
  res.pred = pred;
  res.test_loglik = tll - m;
  res.test_error = pow(val - pred, 2);
  res.cond_pred = cond_pred;
  res.cond_test_loglik = tll - log(cond_norm);
  res.cond_test_error = pow(val - cond_pred, 2);
}

void PMM::update_q_n(unsigned long int ind, double log_lambda, int val,
                     vector<double>& phi, vector<double>& buffer_k,
                     vector<double>& q_n) {
  double lambda = predict(ind, buffer_k);
  for (int n = 1; n < n_trunc; ++n)
    q_n[n - 1] = val * log(lambda) + val * log(phi[n]) - lambda * phi[n] -
                 cache_log_fact[val] - cache_log_fact[n] + n * log_lambda;
  double norm = logsumexp(q_n, n_trunc - 1);
  q_n[0] = 0.0;
  for (int n = n_trunc - 1; n !=0; --n) q_n[n] = exp(q_n[n-1] - norm);
}

void PMM::update(unsigned long int ind, int val, vector<double>& buffer_k,
                 PMMHyperParamStats* stats, double e_phi) {
  double lr = calc_lr(ind);
  calc_varphi(ind, buffer_k);
  update_a_s(ind, lr, val, buffer_k);
  update_b_s(ind, lr, e_phi);

  // TODO Review these updates
  double sum = 0.0;
  for (int j = 0; j < n_component; ++j) sum += b_s[ind][j];
  stats->update_rho_a(n_component * eta);
  stats->update_rho_b(sum);
}

void PMM::update_test_result(unsigned long int ind, double log_lambda, int val,
                             vector<double>& buffer_k, vector<double>& buffer_n,
                             vector<double>& phi, test_result& res) {
  double lambda = predict(ind, buffer_k);
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

UserPMM::UserPMM() {}
UserPMM::UserPMM(unsigned long int n_user_, unsigned long int n_item_,
                 int n_component_, double eta_, double rho_, double xi_,
                 double tau_, int n_trunc_, int val_max_)
    : PMM(n_user_, n_item_, n_component_, eta_, rho_, xi_, tau_, n_trunc_,
          val_max_) {}

void UserPMM::initialize(unsigned long int n_user_, unsigned long int n_item_,
                         int n_component_, double eta_, double rho_, double xi_,
                         double tau_, int n_trunc_, int val_max_) {
  this->PMM::initialize(n_user_, n_item_, n_component_, eta_, rho_, xi_, tau_,
                        n_trunc_, val_max_);
}

void UserPMM::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                            int n_trunc_, double xi_ = 0.7,
                                            double tau_ = 10000) {
  const triplets& ind2train = sm.get_ind2train();
  poisson_params params = mle_poisson(ind2train, 1000);
  double rho_ = 0.1;
  double eta_ = rho_ * params.lambda / (double)n_component_;
  unsigned long int n_item_eff = sm.get_n_item() * (1 - sm.get_sparsity());
  this->PMM::initialize(sm.get_n_user(), n_item_eff, n_component_, eta_, rho_,
                        xi_, tau_, n_trunc_, sm.get_max_response() * 2);
}

void UserPMM::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                            double xi_ = 0.7,
                                            double tau_ = 10000) {
  initialize_with_sparse_matrix(sm, n_component_, 100, xi_, tau_);
}

void UserPMM::serialize(string& fname) { PMM::serialize(fname); }

void UserPMM::deserialize(string& fname) { PMM::deserialize(fname); }

int UserPMM::get_n_trunc() { return PMM::get_n_trunc(); }

int UserPMM::get_n_component() { return PMM::get_n_component(); }

HyperParamStats* UserPMM::create_hyperparam_stats() {
  return PMM::create_hyperparam_stats();
}

void UserPMM::update_hyperparam(HyperParamStats* stats) {
  return PMM::update_hyperparam(dynamic_cast<PMMHyperParamStats*>(stats));
}

double UserPMM::predict(unsigned long int ui, unsigned long int ii,
                        vector<double>& buffer_k) {
  return PMM::predict(ii, buffer_k);
}

void UserPMM::update(unsigned long int ui, unsigned long int ii, double val,
                     vector<double>& buffer_k, vector<double>& buffer_n,
                     HyperParamStats* stats) {
  PMM::update(ii, val, buffer_k, dynamic_cast<PMMHyperParamStats*>(stats));
}

void UserPMM::test(unsigned long int ui, unsigned long int ii, double val,
                   vector<double>& buffer_k, vector<double>& buffer_n,
                   test_result& res) {
  PMM::test(ii, val, buffer_k, res);
}

void UserPMM::update_q_n(unsigned long int ui, unsigned long int ii,
                         double log_lambda, double val, vector<double>& phi,
                         vector<double>& buffer_k, vector<double>& q_n) {
  PMM::update_q_n(ii, log_lambda, val, phi, buffer_k, q_n);
}

void UserPMM::update(unsigned long int ui, unsigned long int ii, double val,
                     vector<double>& buffer_k, vector<double>& buffer_n,
                     HyperParamStats* stats, double e_phi) {
  PMM::update(ii, val, buffer_k, dynamic_cast<PMMHyperParamStats*>(stats),
              e_phi);
}

void UserPMM::update_test_result(unsigned long int ui, unsigned long int ii,
                                 double log_lambda, double val,
                                 vector<double>& buffer_k,
                                 vector<double>& buffer_n, vector<double>& phi,
                                 test_result& res) {
  PMM::update_test_result(ii, log_lambda, val, buffer_k, buffer_n, phi, res);
}

ItemPMM::ItemPMM() {}
ItemPMM::ItemPMM(unsigned long int n_user_, unsigned long int n_item_,
                 int n_component_, double eta_, double rho_, double xi_,
                 double tau_, int n_trunc_, int val_max_)
    : PMM(n_item_, n_user_, n_component_, eta_, rho_, xi_, tau_, n_trunc_,
          val_max_) {}

void ItemPMM::initialize(unsigned long int n_user_, unsigned long int n_item_,
                         int n_component_, double eta_, double rho_, double xi_,
                         double tau_, int n_trunc_, int val_max_) {
  this->PMM::initialize(n_item_, n_user_, n_component_, eta_, rho_, xi_, tau_,
                        n_trunc_, val_max_);
}

void ItemPMM::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                            int n_trunc_, double xi_ = 0.7,
                                            double tau_ = 10000) {
  const triplets& ind2train = sm.get_ind2train();
  poisson_params params = mle_poisson(ind2train, 1000);
  double rho_ = 0.1;
  double eta_ = rho_ * params.lambda / (double)n_component_;
  unsigned long int n_user_eff = sm.get_n_user() * (1 - sm.get_sparsity());
  this->PMM::initialize(sm.get_n_item(), n_user_eff, n_component_, eta_, rho_,
                        xi_, tau_, n_trunc_, sm.get_max_response() * 2);
}

void ItemPMM::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                            double xi_ = 0.7,
                                            double tau_ = 10000) {
  initialize_with_sparse_matrix(sm, n_component_, 100, xi_, tau_);
}

void ItemPMM::serialize(string& fname) { PMM::serialize(fname); }

void ItemPMM::deserialize(string& fname) { PMM::deserialize(fname); }

int ItemPMM::get_n_trunc() { return PMM::get_n_trunc(); }

int ItemPMM::get_n_component() { return PMM::get_n_component(); }

HyperParamStats* ItemPMM::create_hyperparam_stats() {
  return PMM::create_hyperparam_stats();
}

void ItemPMM::update_hyperparam(HyperParamStats* stats) {
  return PMM::update_hyperparam(dynamic_cast<PMMHyperParamStats*>(stats));
}

double ItemPMM::predict(unsigned long int ui, unsigned long int ii,
                        vector<double>& buffer_k) {
  return PMM::predict(ii, buffer_k);
}

void ItemPMM::update(unsigned long int ui, unsigned long int ii, double val,
                     vector<double>& buffer_k, vector<double>& buffer_n,
                     HyperParamStats* stats) {
  PMM::update(ii, val, buffer_k, dynamic_cast<PMMHyperParamStats*>(stats));
}

void ItemPMM::test(unsigned long int ui, unsigned long int ii, double val,
                   vector<double>& buffer_k, vector<double>& buffer_n,
                   test_result& res) {
  PMM::test(ii, val, buffer_k, res);
}

void ItemPMM::update_q_n(unsigned long int ui, unsigned long int ii,
                         double log_lambda, double val, vector<double>& phi,
                         vector<double>& buffer_k, vector<double>& q_n) {
  PMM::update_q_n(ii, log_lambda, val, phi, buffer_k, q_n);
}

void ItemPMM::update(unsigned long int ui, unsigned long int ii, double val,
                     vector<double>& buffer_k, vector<double>& buffer_n,
                     HyperParamStats* stats, double e_phi) {
  PMM::update(ii, val, buffer_k, dynamic_cast<PMMHyperParamStats*>(stats),
              e_phi);
}

void ItemPMM::update_test_result(unsigned long int ui, unsigned long int ii,
                                 double log_lambda, double val,
                                 vector<double>& buffer_k,
                                 vector<double>& buffer_n, vector<double>& phi,
                                 test_result& res) {
  PMM::update_test_result(ii, log_lambda, val, buffer_k, buffer_n, phi, res);
}
