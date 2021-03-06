#ifndef ZTP_H
#define ZTP_H
#include "utils.h"
#include "base.h"

using namespace std;

class ZTPHyperParamStats : public HyperParamStats {
 public:
  ZTPHyperParamStats();
  void initialize(double lambda_0);
  void update_lambda_a(double sum_x);
  void update_lambda_b(double sum_n);
  double get_lambda_gradient();
  void clean(){};

 protected:
  double lambda_a;
  double lambda_b;
};

class ZTP : public HPFCouplingInterface {
 public:
  ZTP();
  ZTP(double lambda_, int n_trunc_, int val_max_, double xi_, double tau_);
  void initialize(double lambda_, int n_trunc_, int val_max_, double xi_,
                  double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_trunc_, double xi_,
                                     double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     int n_trunc_, double xi_, double tau_);
  void serialize(string&);
  void deserialize(string&);
  double get_lambda();
  int get_n_trunc();
  int get_n_component();
  void set_lambda(double lambda_);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(ZTPHyperParamStats* stats);
  void update_hyperparam(HyperParamStats* stats);
  double predict();
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  void update(int val, ZTPHyperParamStats* stats);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats);
  void test(int val, test_result& res);
  void test(unsigned long int ui, unsigned long int ii, double val,
            vector<double>& buffer_k, vector<double>& buffer_n,
            test_result& res);
  void update_q_n(double log_lambda, int val, vector<double>& q_n);
  void update_q_n(unsigned long int ui, unsigned long int ii, double log_lambda,
                  double val, vector<double>& phi, vector<double>& buffer_k,
                  vector<double>& q_n);
  void update(int val, ZTPHyperParamStats* stats, double e_phi);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats, double e_phi);
  void update_test_result(double log_lambda, int val, vector<double>& buffer_n,
                          test_result& res);
  void update_test_result(unsigned long int ui, unsigned long int ii,
                          double log_lambda, double val,
                          vector<double>& buffer_k, vector<double>& buffer_n,
                          vector<double>& phi, test_result& res);

 protected:
  double lambda;
  double lambda_0;
  double element_mean;
  int n_trunc;
  int val_max;
  double xi;
  double tau;
  unsigned long int t_hyper;
  vector<double> cache_log_fact;
  vector<vector<double>> cache_qvar;
};

#endif  // ZTP_H
