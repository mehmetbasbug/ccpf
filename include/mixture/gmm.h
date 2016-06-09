#ifndef GMM_H
#define GMM_H
#include "utils.h"
#include "base.h"

class GMMHyperParamStats : public HyperParamStats {
 public:
  GMMHyperParamStats();
  void initialize(double sigma2_0, double rho_0);
  void update_rho_alpha(double x);
  void update_rho_beta(double x);
  void update_sigma2_alpha(double x);
  void update_sigma2_beta(double x);
  double get_rho_gradient();
  double get_sigma2_gradient();
  void clean(){};

 protected:
  double rho_alpha;
  double rho_beta;
  double sigma2_alpha;
  double sigma2_beta;
};

class GMM {
 public:
  GMM();
  GMM(unsigned long int dim1_, unsigned long int dim2_, int n_component_,
      double eta_, double rho_, double sigma2_,  double xi_, double tau_, int n_trunc_);
  void initialize(unsigned long int dim1_, unsigned long int dim2_,
                  int n_component_, double eta_, double rho_, double sigma2_,
                  double xi_, double tau_, int n_trunc_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     int n_trunc_, double xi_, double tau_);
  void serialize(string& fname);
  void deserialize(string& fname);
  double get_eta();
  double get_eta_0();
  double get_rho();
  double get_rho_0();
  double get_sigma2();
  double get_sigma2_0();
  int get_n_trunc();
  int get_n_component();
  void set_eta(double eta_);
  void set_eta_0(double eta_0_);
  void set_rho(double rho_);
  void set_rho_0(double rho_0_);
  void set_sigma2(double sigma2_);
  void set_sigma2_0(double sigma2_0_);
  double calc_lr(unsigned long int ind);
  void update_a_s(unsigned long int ind, double lr, double err, double phi_mean,
                  double phi_var);
  void update_b_s(unsigned long int ind, double lr, double phi_mean,
                  double phi_var);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(GMMHyperParamStats* stats);
  double predict(unsigned long int ind);
  void update(unsigned long int ind, double val, GMMHyperParamStats* stats);
  void test(unsigned long int ind, double val, test_result& res);
  void update_q_n(unsigned long int ind, double log_lambda, double val,
                  vector<double>& phi_mean, vector<double>& phi_var,
                  vector<double>& q_n);
  void update(unsigned long int ind, double val, GMMHyperParamStats* stats,
              double e_phi_mean, double e_phi_var);
  void update_test_result(unsigned long int ind, double log_lambda, double val,
                          vector<double>& buffer_n, vector<double>& phi_mean,
                          vector<double>& phi_var, test_result& res);

 protected:
  unsigned long int dim1;
  unsigned long int dim2;
  int n_component;
  double eta;
  double eta_0;
  double rho;
  double rho_0;
  double sigma2;
  double sigma2_0;
  int n_trunc;
  double xi;
  double tau;
  unsigned long int t_hyper;
  vector<unsigned long int> t_learning;
  array2d a_s;
  array2d b_s;
  vector<double> cache_log_fact;
};

class UserGMM : public GMM, public HPFNormalCouplingInterface {
 public:
  UserGMM();
  UserGMM(unsigned long int n_user_, unsigned long int n_item_,
          int n_component_, double eta_, double rho_, double sigma2_,
          double xi_, double tau_, int n_trunc_);
  void initialize(unsigned long int n_user_, unsigned long int n_item_,
                  int n_component_, double eta_, double rho_, double sigma2_,
                  double xi_, double tau_, int n_trunc_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     int n_trunc_, double xi_, double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     double xi_, double tau_);
  void serialize(string& fname);
  void deserialize(string& fname);
  int get_n_trunc();
  int get_n_component();
  double get_sigma2();
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
                  double val, vector<double>& phi_mean, vector<double>& phi_var,
                  vector<double>& buffer_k, vector<double>& q_n);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats, double e_phi_mean, double e_phi_var);
  void update_test_result(unsigned long int ui, unsigned long int ii,
                          double log_lambda, double val,
                          vector<double>& buffer_k, vector<double>& buffer_n,
                          vector<double>& phi_mean, vector<double>& phi_var,
                          test_result& res);
};

class ItemGMM : public GMM, public HPFNormalCouplingInterface {
 public:
  ItemGMM();
  ItemGMM(unsigned long int n_user_, unsigned long int n_item_,
          int n_component_, double eta_, double rho_, double sigma2_,
          double xi_, double tau_, int n_trunc_);
  void initialize(unsigned long int n_user_, unsigned long int n_item_,
                  int n_component_, double eta_, double rho_, double sigma2_,
                  double xi_, double tau_, int n_trunc_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     int n_trunc_, double xi_, double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     double xi_, double tau_);
  void serialize(string& fname);
  void deserialize(string& fname);
  int get_n_trunc();
  int get_n_component();
  double get_sigma2();
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
                  double val, vector<double>& phi_mean, vector<double>& phi_var,
                  vector<double>& buffer_k, vector<double>& q_n);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats, double e_phi_mean, double e_phi_var);
  void update_test_result(unsigned long int ui, unsigned long int ii,
                          double log_lambda, double val,
                          vector<double>& buffer_k, vector<double>& buffer_n,
                          vector<double>& phi_mean, vector<double>& phi_var,
                          test_result& res);
};

#endif  // GMM_H
