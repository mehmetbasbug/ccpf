#ifndef INVERSEGAUSSIAN_H
#define INVERSEGAUSSIAN_H
#include "utils.h"
#include "base.h"

using namespace std;

class InverseGaussian : public HPFCouplingInterface {
 public:
  InverseGaussian();
  InverseGaussian(double mu_, double lambda_, int n_trunc_, double xi_,
                  double tau_);
  void initialize(double mu_, double lambda_, int n_trunc_, double xi_,
                  double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_trunc_, double xi_,
                                     double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     int n_trunc_, double xi_, double tau_);
  void serialize(string&);
  void deserialize(string&);
  double get_mu();
  double get_lambda();
  int get_n_trunc();
  int get_n_component();
  void set_mu(double mu_);
  void set_lambda(double lambda_);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(HyperParamStats* stats);
  double predict();
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
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
  double mu;
  double mu_0;
  double lambda;
  double lambda_0;
  double element_mean;
  int n_trunc;
  double xi;
  double tau;
  unsigned long int t_hyper;
  vector<double> cache_log_fact;
};

#endif  // INVERSEGAUSSIAN_H
