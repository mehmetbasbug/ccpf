#ifndef BINOMIAL_H
#define BINOMIAL_H
#include "utils.h"
#include "base.h"

using namespace std;

class BinomialHyperParamStats : public HyperParamStats {
 public:
  BinomialHyperParamStats();
  void initialize(double p_0);
  void update_p_alpha(double sum_x);
  void update_p_beta(double sum_x, double sum_n);
  double get_p_gradient();
  void clean(){};

 private:
  double p_alpha;
  double p_beta;
};

class Binomial : public HPFCouplingInterface {
 public:
  Binomial();
  Binomial(double p_, int r_, int n_trunc_, int val_max_, double xi_,
           double tau_);
  void initialize(double p_, int r_, int n_trunc_, int val_max_, double xi_,
                  double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_trunc_, double xi_,
                                     double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     int n_trunc_, double xi_, double tau_);
  void serialize(string&);
  void deserialize(string&);
  double get_p();
  double get_r();
  int get_n_trunc();
  int get_n_component();
  void set_p(double p_);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(BinomialHyperParamStats* stats);
  void update_hyperparam(HyperParamStats* stats);
  double predict();
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  void update(int val, BinomialHyperParamStats* stats);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats);
  void test(int val, test_result& res);
  void test(unsigned long int ui, unsigned long int ii, double val,
            vector<double>& buffer_k, vector<double>& buffer_n,
            test_result& res);
  void update_q_n(double log_lambda, int val, vector<double>& phi,
                  vector<double>& q_n);
  void update_q_n(unsigned long int ui, unsigned long int ii, double log_lambda,
                  double val, vector<double>& phi, vector<double>& buffer_k,
                  vector<double>& q_n);
  void update(int val, BinomialHyperParamStats* stats, double e_phi);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats, double e_phi);
  void update_test_result(double log_lambda, int val, vector<double>& buffer_n,
                          vector<double>& phi, test_result& res);
  void update_test_result(unsigned long int ui, unsigned long int ii,
                          double log_lambda, double val,
                          vector<double>& buffer_k, vector<double>& buffer_n,
                          vector<double>& phi, test_result& res);

 private:
  double p;
  double p_0;
  int r;
  double element_mean;
  int n_trunc;
  int val_max;
  double xi;
  double tau;
  unsigned long int t_hyper;
  vector<double> cache_log_fact;
  vector<vector<double>> cache_qvar;
};

#endif  // BINOMIAL_H
