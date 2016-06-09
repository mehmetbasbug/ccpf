#ifndef GAUSSIAN_H
#define GAUSSIAN_H
#include "utils.h"
#include "base.h"

using namespace std;

class GaussianHyperParamStats : public HyperParamStats {
 public:
  GaussianHyperParamStats();
  void initialize(double mu_0, double sigma2_0);
  void update_sum_n(double n);
  void update_sum_phi(double e_phi1sq_div_phi2);
  void collapsed_update_sum_phi(double phi1, double phi2);
  void update_sum_x(double val, double e_phi1_div_phi2);
  void collapsed_update_sum_x(double val, double phi1, double phi2);
  void update_sum_x2(double val, double e_1_div_phi2);
  void collapsed_update_sum_x2(double val, double phi2);
  double get_mu_gradient();
  double get_sigma2_gradient(double a, double b);
  double get_a_gradient();
  double get_b_gradient(double mu, double sigma2);
  void clean(){};

 protected:
  double mu_0;
  double lambda_0;
  double a_0;
  double b_0;
  double sum_n;
  double sum_phi;
  double sum_x;
  double sum_x2;
};

class Gaussian : public HPFNormalCouplingInterface{
 public:
  Gaussian();
  Gaussian(double mu_, double sigma2_, int n_trunc_, double xi_, double tau_);
  void initialize(double mu_, double sigma2_, int n_trunc_, double xi_,
                  double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_trunc_, double xi_,
                                     double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     int n_trunc_, double xi_, double tau_);
  void serialize(string&);
  void deserialize(string&);
  double get_mu();
  double get_sigma2();
  int get_n_trunc();
  int get_n_component();
  void set_mu(double mu_);
  void set_sigma2(double a_, double b_);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(GaussianHyperParamStats* stats);
  void update_hyperparam(HyperParamStats* stats);
  double predict();
  double predict(unsigned long int ui, unsigned long int ii, vector<double>& buffer_k);
  void update(double val, GaussianHyperParamStats* stats);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats);
  void test(double val, test_result& res);
  void test(unsigned long int ui, unsigned long int ii, double val,
            vector<double>& buffer_k, vector<double>& buffer_n,
            test_result& res);
  void update_q_n(double log_lambda, double val, vector<double>& phi_mean,
                  vector<double>& phi_var, vector<double>& q_n);
  void update_q_n(unsigned long int ui, unsigned long int ii, double log_lambda,
                  double val, vector<double>& phi_mean, vector<double>& phi_var,
                  vector<double>& buffer_k, vector<double>& q_n);
  void update(double val, GaussianHyperParamStats* stats, double e_phi_mean,
              double e_phi_var);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats, double e_phi_mean, double e_phi_var);
  void update_test_result(double log_lambda, double val, vector<double>& buffer_n,
                          vector<double>& phi_mean, vector<double>& phi_var,
                          test_result& res);
  void update_test_result(unsigned long int ui, unsigned long int ii,
                          double log_lambda, double val,
                          vector<double>& buffer_k, vector<double>& buffer_n,
                          vector<double>& phi_mean, vector<double>& phi_var,
                          test_result& res);

 protected:
  double mu;
  double mu_0;
  double sigma2;
  double sigma2_0;
  double rho_a;
  double rho_b;
  double mu_mu;
  double mu_sigma2;
  double element_mean;
  int n_trunc;
  double xi;
  double tau;
  unsigned long int t_hyper;
  vector<double> cache_log_fact;
};

#endif  // GAUSSIAN_H
