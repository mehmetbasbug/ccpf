#ifndef GAMMA_H
#define GAMMA_H
#include "utils.h"
#include "base.h"

using namespace std;

class GammaHyperParamStats : public HyperParamStats {
 public:
  GammaHyperParamStats();
  void initialize(double beta_0);
  void update_beta_a(double sum_alpha);
  void update_beta_b(double sum_x);
  double get_beta_gradient();
  void clean(){};

 private:
  double beta_a;
  double beta_b;
};

class Gamma : public HPFCouplingInterface {
 public:
  Gamma();
  Gamma(double alpha_, double beta_, int n_trunc_, double xi_, double tau_);
  void initialize(double alpha_, double beta_, int n_trunc_, double xi_,
                  double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_trunc_, double xi_,
                                     double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     int n_trunc_, double xi_, double tau_);
  void serialize(string&);
  void deserialize(string&);
  double get_alpha();
  double get_beta();
  int get_n_trunc();
  int get_n_component();
  void set_alpha(double alpha_);
  void set_beta(double beta_);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(GammaHyperParamStats* stats);
  void update_hyperparam(HyperParamStats* stats);
  double predict();
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  void update(double val, GammaHyperParamStats* stats);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats);
  void test(double val, test_result& res);
  void test(unsigned long int ui, unsigned long int ii, double val,
            vector<double>& buffer_k, vector<double>& buffer_n,
            test_result& res);
  void update_q_n(double log_lambda, double val, vector<double>& phi,
                  vector<double>& q_n);
  void update_q_n(unsigned long int ui, unsigned long int ii, double log_lambda,
                  double val, vector<double>& phi, vector<double>& buffer_k,
                  vector<double>& q_n);
  void update(double val, GammaHyperParamStats* stats, double e_phi);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats, double e_phi);
  void update_test_result(double log_lambda, double val,
                          vector<double>& buffer_n, vector<double>& phi,
                          test_result& res);
  void update_test_result(unsigned long int ui, unsigned long int ii,
                          double log_lambda, double val,
                          vector<double>& buffer_k, vector<double>& buffer_n,
                          vector<double>& phi, test_result& res);

 protected:
  double alpha;
  double alpha_0;
  double beta;
  double beta_0;
  double element_mean;
  int n_trunc;
  double xi;
  double tau;
  unsigned long int t_hyper;
  vector<double> cache_log_fact;
  vector<double> cache_log_scaled_fact;
};

#endif  // GAMMA_H
