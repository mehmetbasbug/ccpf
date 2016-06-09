#ifndef HPF_H
#define HPF_H
#include "utils.h"
#include "base.h"

using namespace std;

class HPFBase {

 public:
  HPFBase();
  HPFBase(unsigned long int n_user_, unsigned long int n_item_, int n_component_,
          double sparsity_, double eta_, double rho_, double varrho_, double zeta_,
          double omega_, double varpi_, double xi_, double tau_, int n_trunc_,
          int val_max_);
  void initialize(unsigned long int n_user_, unsigned long int n_item_,
                  int n_component_, double sparsity_, double eta_, double rho_,
                  double varrho_, double zeta_, double omega_, double varpi_,
                  double xi_, double tau_, int n_trunc_, int val_max_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     int n_trunc_, double xi_, double tau_);
  void serialize(string& fname);
  void deserialize(string& fname);
  double get_eta();
  double get_eta_0();
  double get_rho();
  double get_rho_0();
  double get_varrho();
  double get_varrho_0();
  double get_zeta();
  double get_zeta_0();
  double get_omega();
  double get_omega_0();
  double get_varpi();
  double get_varpi_0();
  int get_n_trunc();
  int get_n_component();
  void set_eta(double eta_);
  void set_eta_0(double eta_0_);
  void set_rho(double rho_);
  void set_rho_0(double rho_0_);
  void set_varrho(double varrho_);
  void set_varrho_0(double varrho_0_);
  void set_zeta(double zeta_);
  void set_zeta_0(double zeta_0_);
  void set_omega(double omega_);
  void set_omega_0(double omega_0_);
  void set_varpi(double varpi_);
  void set_varpi_0(double varpi_0_);
  void set_n_trunc(int n_trunc_);
  double calc_log_lambda(unsigned long int ui, unsigned long int ii,
                         vector<double>& buffer_k);
  double calc_lru(unsigned long int ui);
  double calc_lri(unsigned long int ii);
  void calc_varphi(unsigned long int ui, unsigned long int ii,
                   vector<double>& varphi);
  void update_a_r(unsigned long int ui, double lru);
  void update_b_r(unsigned long int ui, double lru);
  void update_a_s(unsigned long int ui, double lru, double e_n, vector<double>& varphi);
  void update_a_s_with_zero(unsigned long int ui, double lru);
  void update_b_s(unsigned long int ui, unsigned long int ii, double lru, double phi);
  void update_a_w(unsigned long int ii, double lri);
  void update_b_w(unsigned long int ii, double lri);
  void update_a_v(unsigned long int ii, double lri, double e_n, vector<double>& varphi);
  void update_a_v_with_zero(unsigned long int ii, double lri);
  void update_b_v(unsigned long int ui, unsigned long int ii, double lri, double phi);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(HyperParamStats* stats);
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats);
  void test(unsigned long int ui, unsigned long int ii, double val,
            vector<double>& buffer_k, test_result& res);
  void test(unsigned long int ui, unsigned long int ii, double val,
            vector<double>& buffer_k, vector<double>& buffer_n,
            test_result& res);

 protected:
  unsigned long int n_user;
  unsigned long int n_item;
  double sparsity;
  double n_eff_user;
  double n_eff_item;
  int n_component;
  int n_trunc;
  int val_max;
  double eta;
  double eta_0;
  double rho;
  double rho_0;
  double varrho;
  double varrho_0;
  double zeta;
  double zeta_0;
  double omega;
  double omega_0;
  double varpi;
  double varpi_0;
  double xi;
  double tau;
  unsigned long int t_hyper;
  vector<unsigned long int> t_user;
  vector<unsigned long int> t_item;
  vector<double> a_r;
  vector<double> b_r;
  array2d a_s;
  array2d a_v;
  vector<double> a_w;
  vector<double> b_w;
  array2d b_s;
  array2d b_v;
  vector<double> cache_log_fact;
};

class HPF : public HPFBase, public HPFCouplingInterface {

 public:
  HPF();
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     int n_trunc_, double xi_, double tau_);
  void serialize(string& fname);
  void deserialize(string& fname);
  int get_n_trunc();
  int get_n_component();
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(HyperParamStats* stats);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats);
  void test(unsigned long int ui, unsigned long int ii, double val,
            vector<double>& buffer_k, vector<double>& buffer_n,
            test_result& res);
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  void update_q_n(unsigned long int ui, unsigned long int ii, double log_lambda,
                  int val, vector<double>& phi, vector<double>& buffer_k,
                  vector<double>& q_n);
  void update_q_n(unsigned long int ui, unsigned long int ii, double log_lambda,
                  double val, vector<double>& phi, vector<double>& buffer_k,
                  vector<double>& q_n);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, double e_phi);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats, double e_phi);
  void update_test_result(unsigned long int ui, unsigned long int ii,
                          double log_lambda, int val, vector<double>& buffer_k,
                          vector<double>& buffer_n, vector<double>& phi,
                          test_result& res);
  void update_test_result(unsigned long int ui, unsigned long int ii,
                          double log_lambda, double val,
                          vector<double>& buffer_k, vector<double>& buffer_n,
                          vector<double>& phi,
                          test_result& res);
};

#endif  // HPF_H
