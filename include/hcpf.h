#ifndef HCPF_H
#define HCPF_H
#include "utils.h"
#include "base.h"
#include "factor/hpf.h"

using namespace std;

class HCPF : public HPFBase, public HPFInterface {
 public:
  HCPF();
  HCPF(unsigned long int n_user_, unsigned long int n_item_, int n_component_,
       double sparsity_, double eta_, double rho_, double varrho_, double zeta_,
       double omega_, double varpi_, double xi_, double tau_, int n_trunc_,
       int val_max_);
  void initialize(unsigned long int n_user_, unsigned long int n_item_,
                  int n_component_, double sparsity_, double eta_, double rho_,
                  double varrho_, double zeta_, double omega_, double varpi_,
                  double xi_, double tau_, int n_trunc_, int val_max_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     double xi_, double tau_);
  void serialize(string& fname);
  void deserialize(string& fname);
  int get_n_trunc();
  int get_n_component();
  vector<double>& get_phi();
  double calc_log_lambda(unsigned long int ui, unsigned long int ii,
                         vector<double>& buffer_k);
  void calc_scaling(vector<double>& q_n, double& e_n, double& e_phi);
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(HyperParamStats* stats);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, double e_n);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, double e_n, double e_phi, double e_mu,
              HyperParamStats* stats);

 protected:
  vector<double> phi;
};

class HCPFNormal : public HPFBase, public HPFNormalInterface {
 public:
  HCPFNormal();
  HCPFNormal(unsigned long int n_user_, unsigned long int n_item_,
             int n_component_, double sparsity_, double eta_, double rho_,
             double varrho_, double zeta_, double omega_, double varpi_,
             double xi_, double tau_, int n_trunc_, int val_max_);
  void initialize(unsigned long int n_user_, unsigned long int n_item_,
                  int n_component_, double sparsity_, double eta_, double rho_,
                  double varrho_, double zeta_, double omega_, double varpi_,
                  double xi_, double tau_, int n_trunc_, int val_max_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     double xi_, double tau_);
  void serialize(string& fname);
  void deserialize(string& fname);
  int get_n_trunc();
  int get_n_component();
  vector<double>& get_phi_mean();
  vector<double>& get_phi_var();
  double calc_log_lambda(unsigned long int ui, unsigned long int ii,
                         vector<double>& buffer_k);
  void calc_scaling(vector<double>& q_n, double& e_n, double& e_phi_mean,
                    double& e_phi_var);
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(HyperParamStats* stats);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, double e_n);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, double e_n, double e_phi_mean,
              double e_phi_var, double e_mu, double e_sigma2,
              HyperParamStats* stats);

 private:
  vector<double> phi_mean;
  vector<double> phi_var;
};


class HICPFHyperParamStats : public HyperParamStats {
 public:
  HICPFHyperParamStats();
  void initialize();
  void update_cm_cov(double x);
  void update_cm_var(double x);
  double get_cm_gradient();
  void clean(){};

 private:
  double CM_cov;
  double CM_var;
};

class HICPF : public HPFBase, public HPFInterface {
 public:
  HICPF();
  HICPF(unsigned long int n_user_, unsigned long int n_item_, int n_component_,
        double sparsity_, double eta_, double rho_, double varrho_,
        double zeta_, double omega_, double varpi_, double xi_, double tau_,
        int n_trunc_, int val_max_);
  void initialize(unsigned long int n_user_, unsigned long int n_item_,
                  int n_component_, double sparsity_, double eta_, double rho_,
                  double varrho_, double zeta_, double omega_, double varpi_,
                  double xi_, double tau_, int n_trunc_, int val_max_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     double xi_, double tau_);
  void serialize(string& fname);
  void deserialize(string& fname);
  int get_n_trunc();
  int get_n_component();
  vector<double>& get_phi();
  double calc_log_lambda(unsigned long int ui, unsigned long int ii,
                         vector<double>& buffer_k);
  void calc_scaling(vector<double>& q_n, double& e_n, double& e_phi);
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(HyperParamStats* stats);
  void update_hyperparam(HICPFHyperParamStats* stats);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, double e_n, double e_phi, double e_mu,
              HICPFHyperParamStats* stats);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, double e_n, double e_phi, double e_mu,
              HyperParamStats* stats);

 private:
  double cm;
  vector<double> phi;
};

class HICPFNormalHyperParamStats : public HyperParamStats {
 public:
  HICPFNormalHyperParamStats();
  void initialize();
  void update_cm_cov(double x);
  void update_cm_var(double x);
  void update_cv_cov(double x);
  void update_cv_var(double x);
  double get_cm_gradient();
  double get_cv_gradient();
  void clean(){};

 private:
  double CM_cov;
  double CM_var;
  double CV_cov;
  double CV_var;
};

class HICPFNormal : public HPFBase, public HPFNormalInterface {
 public:
  HICPFNormal();
  HICPFNormal(unsigned long int n_user_, unsigned long int n_item_,
              int n_component_, double sparsity_, double eta_, double rho_,
              double varrho_, double zeta_, double omega_, double varpi_,
              double xi_, double tau_, int n_trunc_, int val_max_);
  void initialize(unsigned long int n_user_, unsigned long int n_item_,
                  int n_component_, double sparsity_, double eta_, double rho_,
                  double varrho_, double zeta_, double omega_, double varpi_,
                  double xi_, double tau_, int n_trunc_, int val_max_);
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                     double xi_, double tau_);
  void serialize(string& fname);
  void deserialize(string& fname);
  int get_n_trunc();
  int get_n_component();
  vector<double>& get_phi_mean();
  vector<double>& get_phi_var();
  double calc_log_lambda(unsigned long int ui, unsigned long int ii,
                         vector<double>& buffer_k);
  void calc_scaling(vector<double>& q_n, double& e_n, double& e_phi_mean,
                    double& e_phi_var);
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(HyperParamStats* stats);
  void update_hyperparam(HICPFNormalHyperParamStats* stats);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, double e_n, double e_phi_mean,
              double e_phi_var, double e_mu, double e_sigma2,
              HICPFNormalHyperParamStats* stats);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, double e_n, double e_phi_mean,
              double e_phi_var, double e_mu, double e_sigma2,
              HyperParamStats* stats);

 private:
  double cv;
  double cm;
  vector<double> phi_mean;
  vector<double> phi_var;
};

#endif  // HCPF_H
