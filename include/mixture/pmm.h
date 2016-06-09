#ifndef PMM_H
#define PMM_H
#include "utils.h"
#include "base.h"

class PMMHyperParamStats : public HyperParamStats {
 public:
  PMMHyperParamStats();
  void initialize(double);
  void update_rho_a(double);
  void update_rho_b(double);
  double get_rho_gradient();
  void clean(){};

 protected:
  double rho_a;
  double rho_b;
};

class PMM {
 public:
  PMM();
  PMM(unsigned long int dim1_, unsigned long int dim2_, int n_component_,
      double eta_, double rho_, double xi_, double tau_, int n_trunc_,
      int val_max_);
  void initialize(unsigned long int dim1_, unsigned long int dim2_,
                  int n_component_, double eta_, double rho_, double xi_,
                  double tau_, int n_trunc_, int val_max_);
  void serialize(string& fname);
  void deserialize(string& fname);
  double get_eta();
  double get_eta_0();
  double get_rho();
  double get_rho_0();
  int get_n_trunc();
  int get_n_component();
  void set_eta(double eta_);
  void set_eta_0(double eta_0_);
  void set_rho(double rho_);
  void set_rho_0(double rho_0_);
  double calc_lr(unsigned long int ind);
  void calc_varphi(unsigned long int ind, vector<double>& varphi);
  void update_a_s(unsigned long int ind, double lr, double val,
                  vector<double>& varphi);
  void update_b_s(unsigned long int ind, double lr, double phi);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(PMMHyperParamStats* stats);
  double predict(unsigned long int ind, vector<double>& buffer_k);
  void update(unsigned long int ind, int val, vector<double>& buffer_k,
              PMMHyperParamStats* stats);
  void test(unsigned long int ind, int val, vector<double>& buffer_k,
            test_result& res);
  void update_q_n(unsigned long int ind, double log_lambda, int val,
                  vector<double>& phi, vector<double>& buffer_k,
                  vector<double>& q_n);
  void update(unsigned long int ind, int val, vector<double>& buffer_k,
              PMMHyperParamStats* stats, double e_phi);
  void update_test_result(unsigned long int ind, double log_lambda, int val,
                          vector<double>& buffer_k, vector<double>& buffer_n,
                          vector<double>& phi, test_result& res);

 protected:
  unsigned long int dim1;
  unsigned long int dim2;
  int n_component;
  double eta;
  double eta_0;
  double rho;
  double rho_0;
  double xi;
  double tau;
  int n_trunc;
  int val_max;
  unsigned long int t_hyper;
  vector<unsigned long int> t_learning;
  array2d a_s;
  array2d b_s;
  vector<double> cache_log_fact;
};

class UserPMM : public PMM, public HPFCouplingInterface {
 public:
  UserPMM();
  UserPMM(unsigned long int n_user_, unsigned long int n_item_,
          int n_component_, double eta_, double rho_, double xi_, double tau_,
          int n_trunc_, int val_max_);
  void initialize(unsigned long int n_user_, unsigned long int n_item_,
                  int n_component_, double eta_, double rho_, double xi_,
                  double tau_, int n_trunc_, int val_max_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     int n_trunc_, double xi_, double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     double xi_, double tau_);
  void serialize(string& fname);
  void deserialize(string& fname);
  int get_n_trunc();
  int get_n_component();
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(HyperParamStats* stats);
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats);
  void test(unsigned long int ui, unsigned long int ii, double val,
            vector<double>& buffer_k, vector<double>& buffer_n,
            test_result& res);
  void update_q_n(unsigned long int ui, unsigned long int ii, double log_lambda,
                  double val, vector<double>& phi, vector<double>& buffer_k,
                  vector<double>& q_n);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats, double e_phi);
  void update_test_result(unsigned long int ui, unsigned long int ii,
                          double log_lambda, double val,
                          vector<double>& buffer_k, vector<double>& buffer_n,
                          vector<double>& phi, test_result& res);
};

class ItemPMM : public PMM, public HPFCouplingInterface {
 public:
  ItemPMM();
  ItemPMM(unsigned long int n_user_, unsigned long int n_item_,
          int n_component_, double eta_, double rho_, double xi_, double tau_,
          int n_trunc_, int val_max_);
  void initialize(unsigned long int n_user_, unsigned long int n_item_,
                  int n_component_, double eta_, double rho_, double xi_,
                  double tau_, int n_trunc_, int val_max_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     int n_trunc_, double xi_, double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     double xi_, double tau_);
  void serialize(string& fname);
  void deserialize(string& fname);
  int get_n_trunc();
  int get_n_component();
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(HyperParamStats* stats);
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats);
  void test(unsigned long int ui, unsigned long int ii, double val,
            vector<double>& buffer_k, vector<double>& buffer_n,
            test_result& res);
  void update_q_n(unsigned long int ui, unsigned long int ii, double log_lambda,
                  double val, vector<double>& phi, vector<double>& buffer_k,
                  vector<double>& q_n);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats, double e_phi);
  void update_test_result(unsigned long int ui, unsigned long int ii,
                          double log_lambda, double val,
                          vector<double>& buffer_k, vector<double>& buffer_n,
                          vector<double>& phi, test_result& res);
};

#endif  // PMM_H
