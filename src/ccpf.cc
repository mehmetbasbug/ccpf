#include "utils.h"
#include "ccpf.h"
using namespace std;
namespace bmath = boost::math;

CCPFHyperParamStats::CCPFHyperParamStats() {}

HyperParamStats* CCPFHyperParamStats::get_rmdl_hyperparamstats() {
  return rmdlhps;
}

HyperParamStats* CCPFHyperParamStats::get_smdl_hyperparamstats() {
  return smdlhps;
}

void CCPFHyperParamStats::set_rmdl_hyperparamstats(HyperParamStats* rmdlhps_) {
  rmdlhps = rmdlhps_;
}

void CCPFHyperParamStats::set_smdl_hyperparamstats(HyperParamStats* smdlhps_) {
  smdlhps = smdlhps_;
}

void CCPFHyperParamStats::clean() {
  delete smdlhps;
  delete rmdlhps;
}

CCPF::CCPF() {}

void CCPF::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_, int n_trunc_,
                                         double xi_ = 0.7,
                                         double tau_ = 10000) {
  smdl->initialize_with_sparse_matrix(sm, n_component_, xi_, tau_);
  n_trunc_ = smdl->get_n_trunc();
  rmdl->initialize_with_sparse_matrix(sm, n_component_, n_trunc_, xi_, tau_);
}

void CCPF::serialize(string& fname) {
  string rm_fname;
  rm_fname = fname;
  rm_fname.insert(rm_fname.length() - 3, "_rm");
  rmdl->serialize(rm_fname);
  string sm_fname;
  sm_fname = fname;
  sm_fname.insert(sm_fname.length() - 3, "_sm");
  smdl->serialize(sm_fname);
}

void CCPF::deserialize(string& fname) {
  string rm_fname;
  rm_fname = fname;
  rm_fname.insert(rm_fname.length() - 3, "_rm");
  rmdl->deserialize(rm_fname);
  string sm_fname;
  sm_fname = fname;
  sm_fname.insert(sm_fname.length() - 3, "_sm");
  smdl->deserialize(sm_fname);
}

int CCPF::get_n_trunc() { return smdl->get_n_trunc(); }

int CCPF::get_n_component() { return smdl->get_n_component(); }

HPFCouplingInterface* CCPF::get_rmdl() { return rmdl; }

HPFInterface* CCPF::get_smdl() { return smdl; }

void CCPF::set_rmdl(HPFCouplingInterface* rmdl_) { rmdl = rmdl_; }

void CCPF::set_smdl(HPFInterface* smdl_) { smdl = smdl_; }

void CCPF::update(unsigned long int ui, unsigned long int ii, double val,
                  vector<double>& buffer_k, vector<double>& buffer_n,
                  HyperParamStats* stats) {
  update(ui, ii, val, buffer_k, buffer_n,
         dynamic_cast<CCPFHyperParamStats*>(stats));
}

void CCPF::update(unsigned long int ui, unsigned long int ii, double val,
                  vector<double>& buffer_k, vector<double>& buffer_n,
                  CCPFHyperParamStats* stats) {
  if (val == 0) {
    HyperParamStats* smdlhps = stats->get_smdl_hyperparamstats();
    smdl->update(ui, ii, val, buffer_k, 1.0, 1.0, 0.0, smdlhps);
  } else {
    double log_lambda = smdl->calc_log_lambda(ui, ii, buffer_k);
    vector<double>& phi = smdl->get_phi();
    rmdl->update_q_n(ui, ii, log_lambda, val, phi, buffer_k, buffer_n);
    double e_phi, e_n;
    smdl->calc_scaling(buffer_n, e_n, e_phi);
    HyperParamStats* rmdlhps = stats->get_rmdl_hyperparamstats();
    HyperParamStats* smdlhps = stats->get_smdl_hyperparamstats();
    double e_mu = rmdl->predict(ui,ii,buffer_k);
    smdl->update(ui, ii, val, buffer_k, e_n, e_phi, e_mu, smdlhps);
    rmdl->update(ui, ii, val, buffer_k, buffer_n, rmdlhps, e_phi);
  }
}

void CCPF::test(unsigned long int ui, unsigned long int ii, double val,
                vector<double>& buffer_k, vector<double>& buffer_n,
                test_result& res) {
  double log_lambda = smdl->calc_log_lambda(ui, ii, buffer_k);
  vector<double>& phi = smdl->get_phi();
  rmdl->update_test_result(ui, ii, log_lambda, val, buffer_k, buffer_n,
                           phi, res);
}

double CCPF::predict(unsigned long int ui, unsigned long int ii,
                     vector<double>& buffer_k) {
  double rmdlm = rmdl->predict(ui, ii, buffer_k);
  double smdlm = smdl->predict(ui, ii, buffer_k);
  return rmdlm * smdlm;
}

HyperParamStats* CCPF::create_hyperparam_stats() {
  HyperParamStats* rmdlhps = rmdl->create_hyperparam_stats();
  HyperParamStats* smdlhps = smdl->create_hyperparam_stats();
  CCPFHyperParamStats* hps = new CCPFHyperParamStats();
  hps->set_rmdl_hyperparamstats(rmdlhps);
  hps->set_smdl_hyperparamstats(smdlhps);
  return hps;
}

void CCPF::update_hyperparam(HyperParamStats* stats) {
  update_hyperparam(dynamic_cast<CCPFHyperParamStats*>(stats));
}

void CCPF::update_hyperparam(CCPFHyperParamStats* stats) {
  HyperParamStats* rmdlhps = stats->get_rmdl_hyperparamstats();
  HyperParamStats* smdlhps = stats->get_smdl_hyperparamstats();
  rmdl->update_hyperparam(rmdlhps);
  smdl->update_hyperparam(smdlhps);
}

CCPFNormal::CCPFNormal() {}

void CCPFNormal::initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_, int n_trunc_,
                                               double xi_ = 0.7, double tau_ = 10000) {
  smdl->initialize_with_sparse_matrix(sm, n_component_, xi_, tau_);
  n_trunc_ = smdl->get_n_trunc();
  rmdl->initialize_with_sparse_matrix(sm, n_component_, n_trunc_, xi_, tau_);
}

void CCPFNormal::serialize(string& fname) {
  string rm_fname;
  rm_fname = fname;
  rm_fname.insert(rm_fname.length() - 3, "_rm");
  rmdl->serialize(rm_fname);
  string sm_fname;
  sm_fname = fname;
  sm_fname.insert(sm_fname.length() - 3, "_sm");
  smdl->serialize(sm_fname);
}

void CCPFNormal::deserialize(string& fname) {
  string rm_fname;
  rm_fname = fname;
  rm_fname.insert(rm_fname.length() - 3, "_rm");
  rmdl->deserialize(rm_fname);
  string sm_fname;
  sm_fname = fname;
  sm_fname.insert(sm_fname.length() - 3, "_sm");
  smdl->deserialize(sm_fname);
}

int CCPFNormal::get_n_trunc() { return smdl->get_n_trunc(); }

int CCPFNormal::get_n_component() { return smdl->get_n_component(); }

double CCPFNormal::get_sigma2() { return rmdl->get_sigma2(); }

HPFNormalCouplingInterface* CCPFNormal::get_rmdl() { return rmdl; }

HPFNormalInterface* CCPFNormal::get_smdl() { return smdl; }

void CCPFNormal::set_rmdl(HPFNormalCouplingInterface* rmdl_) { rmdl = rmdl_; }

void CCPFNormal::set_smdl(HPFNormalInterface* smdl_) { smdl = smdl_; }

void CCPFNormal::update(unsigned long int ui, unsigned long int ii, double val,
                        vector<double>& buffer_k, vector<double>& buffer_n,
                        HyperParamStats* stats) {
  update(ui, ii, val, buffer_k, buffer_n,
         dynamic_cast<CCPFHyperParamStats*>(stats));
}

void CCPFNormal::update(unsigned long int ui, unsigned long int ii, double val,
                        vector<double>& buffer_k, vector<double>& buffer_n,
                        CCPFHyperParamStats* stats) {
  if (val == 0) {
    HyperParamStats* smdlhps = stats->get_smdl_hyperparamstats();
    smdl->update(ui, ii, val, buffer_k, 1.0, 1.0, 1.0, 0.0, 0.0, smdlhps);
  } else {
    double log_lambda = smdl->calc_log_lambda(ui, ii, buffer_k);
    vector<double>& phi_mean = smdl->get_phi_mean();
    vector<double>& phi_var = smdl->get_phi_var();
    rmdl->update_q_n(ui, ii, log_lambda, val, phi_mean, phi_var,
                     buffer_k, buffer_n);
    double e_phi_mean, e_phi_var, e_n;
    smdl->calc_scaling(buffer_n, e_n, e_phi_mean, e_phi_var);
    HyperParamStats* rmdlhps = stats->get_rmdl_hyperparamstats();
    HyperParamStats* smdlhps = stats->get_smdl_hyperparamstats();
    double e_mu = rmdl->predict(ui,ii,buffer_k);
    double e_sigma2 = rmdl->get_sigma2();
    smdl->update(ui, ii, val, buffer_k, e_n, e_phi_mean, e_phi_var, e_mu,
                 e_sigma2, smdlhps);
    rmdl->update(ui, ii, val, buffer_k, buffer_n, rmdlhps, e_phi_mean,
                 e_phi_var);
  }
}

void CCPFNormal::test(unsigned long int ui, unsigned long int ii, double val,
                      vector<double>& buffer_k, vector<double>& buffer_n,
                      test_result& res) {
  double log_lambda = smdl->calc_log_lambda(ui, ii, buffer_k);
  vector<double>& phi_mean = smdl->get_phi_mean();
  vector<double>& phi_var = smdl->get_phi_var();
  rmdl->update_test_result(ui, ii, log_lambda, val, buffer_k, buffer_n,
                           phi_mean, phi_var, res);
}

double CCPFNormal::predict(unsigned long int ui, unsigned long int ii,
                           vector<double>& buffer_k) {
  double rmdlm = rmdl->predict(ui, ii, buffer_k);
  double smdlm = smdl->predict(ui, ii, buffer_k);
  return rmdlm * smdlm;
}

HyperParamStats* CCPFNormal::create_hyperparam_stats() {
  HyperParamStats* rmdlhps = rmdl->create_hyperparam_stats();
  HyperParamStats* smdlhps = smdl->create_hyperparam_stats();
  CCPFHyperParamStats* hps = new CCPFHyperParamStats();
  hps->set_rmdl_hyperparamstats(rmdlhps);
  hps->set_smdl_hyperparamstats(smdlhps);
  return hps;
}

void CCPFNormal::update_hyperparam(HyperParamStats* stats) {
  update_hyperparam(dynamic_cast<CCPFHyperParamStats*>(stats));
}

void CCPFNormal::update_hyperparam(CCPFHyperParamStats* stats) {
  HyperParamStats* rmdlhps = stats->get_rmdl_hyperparamstats();
  HyperParamStats* smdlhps = stats->get_smdl_hyperparamstats();
  rmdl->update_hyperparam(rmdlhps);
  smdl->update_hyperparam(smdlhps);
}
