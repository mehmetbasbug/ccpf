#ifndef PMF_H
#define PMF_H
#include "utils.h"
#include "base.h"

using namespace std;

class PMFHyperParamStats : public HyperParamStats {
 public:
  PMFHyperParamStats();
  void initialize(double sigma2_0, double rho_0, double omega_0);
  void update_rho_alpha(double x);
  void update_rho_beta(double x);
  void update_omega_alpha(double x);
  void update_omega_beta(double x);
  void update_sigma2_alpha(double x);
  void update_sigma2_beta(double x);
  double get_rho_gradient();
  double get_omega_gradient();
  double get_sigma2_gradient();
  void clean(){};

 protected:
  double rho_alpha;
  double rho_beta;
  double omega_alpha;
  double omega_beta;
  double sigma2_alpha;
  double sigma2_beta;
};

class PMF : public HPFNormalCouplingInterface {
 public:
  PMF();
  PMF(unsigned long int n_user_, unsigned long int n_item_, int n_component_,
      double sparsity_, double eta_, double rho_, double zeta_, double omega_,
      double sigma2_, double xi_, double tau_, int n_trunc_);
  void initialize(unsigned long int n_user_, unsigned long int n_item_,
                  int n_component_, double sparsity_, double eta_, double rho_,
                  double zeta_, double omega_, double sigma2_, double xi_,
                  double tau_, int n_trunc_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     int n_trunc_, double xi_, double tau_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     double xi_, double tau_);
  void serialize(string& fname);
  void deserialize(string& fname);
  double get_eta();
  double get_eta_0();
  double get_rho();
  double get_rho_0();
  double get_zeta();
  double get_zeta_0();
  double get_omega();
  double get_omega_0();
  double get_sigma2();
  double get_sigma2_0();
  int get_n_trunc();
  int get_n_component();
  void set_eta(double eta_);
  void set_eta_0(double eta_0_);
  void set_rho(double rho_);
  void set_rho_0(double rho_0_);
  void set_zeta(double zeta_);
  void set_zeta_0(double zeta_0_);
  void set_omega(double omega_);
  void set_omega_0(double omega_0_);
  void set_sigma2(double sigma2_);
  void set_sigma2_0(double sigma2_0_);
  double calc_lru(unsigned long int ui);
  double calc_lri(unsigned long int ii);
  void update_a_s(unsigned long int ui, unsigned long int ii, double lru,
                  vector<double>& err, double phi_mean, double phi_var);
  void update_b_s(unsigned long int ui, unsigned long int ii, double lru,
                  double phi_mean, double phi_var);
  void update_a_v(unsigned long int ui, unsigned long int ii, double lru,
                  vector<double>& err, double phi_mean, double phi_var);
  void update_b_v(unsigned long int ui, unsigned long int ii, double lru,
                  double phi_mean, double phi_var);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(PMFHyperParamStats* stats);
  void update_hyperparam(HyperParamStats* stats);
  double predict(unsigned long int ui, unsigned long int ii);
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, PMFHyperParamStats* stats);
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
              vector<double>& buffer_k, PMFHyperParamStats* stats,
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

 protected:
  unsigned long int n_user;
  unsigned long int n_item;
  double sparsity;
  double n_eff_user;
  double n_eff_item;
  int n_component;
  double eta;
  double eta_0;
  double rho;
  double rho_0;
  double zeta;
  double zeta_0;
  double omega;
  double omega_0;
  double sigma2;
  double sigma2_0;
  int n_trunc;
  double xi;
  double tau;
  unsigned long int t_hyper;
  vector<unsigned long int> t_user;
  vector<unsigned long int> t_item;
  array2d a_s;
  array2d b_s;
  array2d a_v;
  array2d b_v;
  vector<double> cache_log_fact;
};

#endif  // PMF_H
