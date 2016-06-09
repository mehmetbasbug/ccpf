#ifndef Regression_H
#define Regression_H
#include "utils.h"
#include "base.h"
using namespace std;

class RegressionHyperParamStats : public HyperParamStats {
 public:
  RegressionHyperParamStats();
  void initialize(double sigma2_0, double rho_0);
  void update_sigma2_alpha(double x);
  void update_sigma2_beta(double x);
  void update_rho_alpha(double x);
  void update_rho_beta(double x);
  double get_sigma2_gradient();
  double get_rho_gradient();
  void clean(){};

 protected:
  double sigma2_alpha;
  double sigma2_beta;
  double rho_alpha;
  double rho_beta;
};

class Regression : public HPFNormalCouplingInterface {
 public:
  Regression();
  Regression(unsigned long int n_user_, unsigned long int n_item_,
             int n_component_, double sparsity_, double rho_, double sigma2_,
             double xi_, double tau_, int n_trunc_);
  void initialize(unsigned long int n_user_, unsigned long int n_item_,
                  int n_component_, double sparsity_, double rho_,
                  double sigma2_, double xi_, double tau_, int n_trunc_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     int n_trunc_, double xi_, double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     double xi_, double tau_);
  void serialize(string& fname);
  void deserialize(string& fname);
  double get_rho();
  double get_rho_0();
  double get_sigma2();
  double get_sigma2_0();
  int get_n_trunc();
  int get_n_component();
  int get_n_xreg();
  void set_n_xreg(int n_xreg_);
  void set_rho(double rho_);
  void set_rho_0(double rho_0_);
  void set_sigma2(double sigma2_);
  void set_sigma2_0(double sigma2_0_);
  double calc_lru(unsigned long int ui);
  void update_a_s(unsigned long int ui, unsigned long int ii, double lru,
                  vector<double>& err, double phi_mean, double phi_var);
  void update_b_s(unsigned long int ui, unsigned long int ii, double lru,
                  double phi_mean, double phi_var);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(RegressionHyperParamStats* stats);
  void update_hyperparam(HyperParamStats* stats);
  double predict(unsigned long int ui, unsigned long int ii);
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, RegressionHyperParamStats* stats);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats);
  void test(unsigned long int ui, unsigned long int ii, double val,
            test_result& res);
  void test(unsigned long int ui, unsigned long int ii, double val,
            vector<double>& buffer_k, vector<double>& buffer_n,
            test_result& res);
  void update_q_n(unsigned long int ui, unsigned long int ii, double log_lambda,
                  double val, vector<double>& phi_mean, vector<double>& phi_var,
                  vector<double>& q_n);
  void update_q_n(unsigned long int ui, unsigned long int ii, double log_lambda,
                  double val, vector<double>& phi_mean, vector<double>& phi_var,
                  vector<double>& buffer_k, vector<double>& q_n);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, RegressionHyperParamStats* stats,
              double e_phi_mean, double e_phi_var);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats, double e_phi_mean, double e_phi_var);
  void update_test_result(unsigned long int ui, unsigned long int ii,
                          double log_lambda, double val,
                          vector<double>& buffer_n, vector<double>& phi_mean,
                          vector<double>& phi_var, test_result& res);
  void update_test_result(unsigned long int ui, unsigned long int ii,
                          double log_lambda, double val,
                          vector<double>& buffer_k, vector<double>& buffer_n,
                          vector<double>& phi_mean, vector<double>& phi_var,
                          test_result& res);
  void load_xreg(string& fname);
  void reset_coef();

 protected:
  unsigned long int n_user;
  unsigned long int n_item;
  double sparsity;
  double n_eff_user;
  double n_eff_item;
  int n_component;
  double rho;
  double rho_0;
  double sigma2;
  double sigma2_0;
  int n_trunc;
  double xi;
  double tau;
  unsigned long int t_hyper;
  vector<unsigned long int> t_learning;
  vector<double> eta;
  array2d a_s;
  array2d b_s;
  array2d xreg;
  vector<double> cache_log_fact;
};

#endif  // Regression_H
