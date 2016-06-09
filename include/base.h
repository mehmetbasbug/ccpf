#ifndef BASE_H
#define BASE_H
#include "utils.h"
#include "sparse_matrix.h"

using namespace std;

class Configuration {
 public:
  Configuration();
  unsigned long int get_conf_num();
  string& get_in_dir();
  string& get_out_dir();
  string& get_response_model();
  string& get_sparsity_model();
  string& get_dataset();
  string& get_sample_method();
  unsigned long int get_batchsize();
  unsigned long int get_test_every();
  unsigned long int get_save_pred_every();
  unsigned long int get_save_every();
  unsigned long int get_max_iter();
  int get_n_component();
  double get_xi();
  double get_tau();
  double get_test_ratio();
  double get_validation_ratio();
  unsigned long int get_max_save_iter();
  void read(string& fname, unsigned long int conf_num_);
  void write_json(bptree::ptree& pt);

 private:
  unsigned long int conf_num;
  string in_dir;
  string out_dir;
  string response_model;
  string sparsity_model;
  string dataset;
  string sample_method;
  unsigned long int batchsize;
  unsigned long int test_every;
  unsigned long int save_pred_every;
  unsigned long int save_every;
  unsigned long int max_iter;
  int n_component;
  double xi;
  double tau;
  double test_ratio;
  double validation_ratio;
};

class Result {
 public:
  Result(Configuration& cfg_, SparseMatrix& sm, int n_trunc_);
  void update(double val_z_tll_sum, double val_nz_tll_sum,
              double val_cond_nz_tll_sum, double val_z_err_sum,
              double val_nz_err_sum, double val_cond_nz_err_sum,
              double z_tll_sum, double nz_tll_sum, double cond_nz_tll_sum,
              double z_err_sum, double nz_err_sum, double cond_nz_err_sum,
              double recall20_sum, double recall50_sum, double recall100_sum,
              double map20_sum, double map50_sum, double map100_sum,
              double ndcg20_sum, double ndcg50_sum, double ndcg100_sum,
              double auc_, unsigned long long int n_eff_user);
  void write_json(bptree::ptree& pt);
  void save();

 private:
  Configuration& cfg;
  unsigned long long int n_user;
  unsigned long long int n_item;
  unsigned long long int n_total_nonzero;
  unsigned long long int n_train_nonzero;
  unsigned long long int n_test_nonzero;
  unsigned long long int n_val_nonzero;
  unsigned long long int n_total_zero;
  unsigned long long int n_train_zero;
  unsigned long long int n_test_zero;
  unsigned long long int n_val_zero;
  unsigned long long int n_eff_test_zero;
  unsigned long long int n_eff_val_zero;
  unsigned long int batchsize;
  unsigned long int test_every;
  unsigned long int save_pred_every;
  unsigned long int save_every;
  int max_save_iter;
  int index;
  int n_trunc;
  double sparsity;
  double max_response;
  double min_response;
  vector<double> val_z_tll;
  vector<double> val_nz_tll;
  vector<double> val_cond_nz_tll;
  vector<double> val_tll;
  vector<double> z_tll;
  vector<double> nz_tll;
  vector<double> cond_nz_tll;
  vector<double> tll;
  vector<double> val_z_err;
  vector<double> val_nz_err;
  vector<double> val_cond_nz_err;
  vector<double> val_err;
  vector<double> z_err;
  vector<double> nz_err;
  vector<double> cond_nz_err;
  vector<double> err;
  vector<double> recall20;
  vector<double> recall50;
  vector<double> recall100;
  vector<double> map20;
  vector<double> map50;
  vector<double> map100;
  vector<double> ndcg20;
  vector<double> ndcg50;
  vector<double> ndcg100;
  vector<double> auc;
};

class HyperParamStats {
 public:
  HyperParamStats(){};
  virtual~HyperParamStats(){};
  virtual void clean() = 0;
};

class NullHyperParamStats : public HyperParamStats {
 public:
  NullHyperParamStats(){};
  void clean(){};
};

class Model {
 public:
  Model(){};
  virtual void initialize_with_sparse_matrix(SparseMatrix& sm, int n_trunc_,
                                             int n_component_,
                                             double xi_, double tau_) = 0;
  virtual void serialize(string& fname) = 0;
  virtual void deserialize(string& fname) = 0;
  virtual int get_n_trunc() = 0;
  virtual int get_n_component() = 0;
  virtual HyperParamStats* create_hyperparam_stats() = 0;
  virtual void update_hyperparam(HyperParamStats* stats) = 0;
  virtual double predict(unsigned long int ui, unsigned long int ii,
                         vector<double>& buffer_k) = 0;
  virtual void update(unsigned long int ui, unsigned long int ii, double val,
                      vector<double>& buffer_k, vector<double>& buffer_n,
                      HyperParamStats* stats) = 0;
  virtual void test(unsigned long int ui, unsigned long int ii, double val,
                    vector<double>& buffer_k, vector<double>& buffer_n,
                    test_result& res) = 0;
  void fit(SparseMatrix& sm, unsigned long int batchsize_,
           unsigned long int max_iter_, unsigned long int test_every_,
           unsigned long int save_every_, unsigned long int save_pred_every_,
           string& sample_method_, string& out_dir_, Result& result,
           unsigned long long int n_user_max_);
};

class CouplingInterface {
 public:
  CouplingInterface(){};
  virtual ~CouplingInterface(){};
};

class HPFCouplingInterface : public Model {
 public:
  HPFCouplingInterface() {}
  virtual void update_q_n(unsigned long int ui, unsigned long int ii,
                          double log_lambda, double val, vector<double>& phi,
                          vector<double>& buffer_k, vector<double>& q_n) = 0;
  virtual void update(unsigned long int ui, unsigned long int ii, double val,
                      vector<double>& buffer_k, vector<double>& buffer_n,
                      HyperParamStats* stats, double e_phi) = 0;
  virtual void update_test_result(unsigned long int ui, unsigned long int ii,
                                  double log_lambda, double val,
                                  vector<double>& buffer_k,
                                  vector<double>& buffer_n, vector<double>& phi,
                                  test_result& res) = 0;
};

class HPFNormalCouplingInterface : public Model {
 public:
  HPFNormalCouplingInterface() {}
  virtual double get_sigma2() = 0;
  virtual void update_q_n(unsigned long int ui, unsigned long int ii,
                          double log_lambda, double val,
                          vector<double>& phi_mean, vector<double>& phi_var,
                          vector<double>& buffer_k, vector<double>& q_n) = 0;
  virtual void update(unsigned long int ui, unsigned long int ii, double val,
                      vector<double>& buffer_k, vector<double>& buffer_n,
                      HyperParamStats* stats, double e_phi_mean,
                      double e_phi_var) = 0;
  virtual void update_test_result(
      unsigned long int ui, unsigned long int ii, double log_lambda, double val,
      vector<double>& buffer_k, vector<double>& buffer_n,
      vector<double>& phi_mean, vector<double>& phi_var, test_result& res) = 0;
};

class HPFInterface {
 public:
  HPFInterface() {}
  virtual void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                             double xi_, double tau_) = 0;
  virtual void serialize(string& fname) = 0;
  virtual void deserialize(string& fname) = 0;
  virtual int get_n_trunc() = 0;
  virtual int get_n_component() = 0;
  virtual vector<double>& get_phi() = 0;
  virtual double calc_log_lambda(unsigned long int ui, unsigned long int ii,
                                 vector<double>& buffer_k) = 0;
  virtual void calc_scaling(vector<double>& q_n, double& e_n,
                            double& e_phi) = 0;
  virtual double predict(unsigned long int ui, unsigned long int ii,
                         vector<double>& buffer_k) = 0;
  virtual HyperParamStats* create_hyperparam_stats() = 0;
  virtual void update_hyperparam(HyperParamStats* stats) = 0;
  virtual void update(unsigned long int ui, unsigned long int ii, double val,
                      vector<double>& buffer_k, double e_n, double e_phi,
                      double e_mu, HyperParamStats* stats) = 0;

 protected:
  vector<double> phi;
};

class HPFNormalInterface {
 public:
  HPFNormalInterface() {}
  virtual void serialize(string& fname) = 0;
  virtual void deserialize(string& fname) = 0;
  virtual void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_,
                                             double xi_, double tau_) = 0;
  virtual int get_n_trunc() = 0;
  virtual int get_n_component() = 0;
  virtual vector<double>& get_phi_mean() = 0;
  virtual vector<double>& get_phi_var() = 0;
  virtual double calc_log_lambda(unsigned long int ui, unsigned long int ii,
                                 vector<double>& buffer_k) = 0;
  virtual void calc_scaling(vector<double>& q_n, double& e_n,
                            double& e_phi_mean, double& e_phi_var) = 0;
  virtual double predict(unsigned long int ui, unsigned long int ii,
                         vector<double>& buffer_k) = 0;
  virtual HyperParamStats* create_hyperparam_stats() = 0;
  virtual void update_hyperparam(HyperParamStats* stats) = 0;
  virtual void update(unsigned long int ui, unsigned long int ii, double val,
                      vector<double>& buffer_k, double e_n, double e_phi_mean,
                      double e_phi_var, double e_mu, double e_sigma2,
                      HyperParamStats* stats) = 0;

 protected:
  vector<double> phi_mean;
  vector<double> phi_var;
};

#endif  // BASE_H
