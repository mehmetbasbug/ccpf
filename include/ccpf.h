#ifndef CCPF_H
#define CCPF_H
#include "utils.h"
#include "base.h"

class CCPFHyperParamStats : public HyperParamStats {
 public:
  CCPFHyperParamStats();
  void initialize();
  HyperParamStats* get_rmdl_hyperparamstats();
  HyperParamStats* get_smdl_hyperparamstats();
  void set_rmdl_hyperparamstats(HyperParamStats* rmdlhps_);
  void set_smdl_hyperparamstats(HyperParamStats* smdlhps_);
  void clean();

 protected:
  HyperParamStats* rmdlhps;
  HyperParamStats* smdlhps;
};

class CCPF : public Model {
 public:
  CCPF();
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_, int n_trunc_,
                                     double xi_, double tau_);
  void serialize(string&);
  void deserialize(string&);
  int get_n_trunc();
  int get_n_component();
  HPFCouplingInterface* get_rmdl();
  HPFInterface* get_smdl();
  void set_rmdl(HPFCouplingInterface* rmdl_);
  void set_smdl(HPFInterface* smdl_);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              CCPFHyperParamStats* stats);
  void test(unsigned long int ui, unsigned long int ii, double val,
            vector<double>& buffer_k, vector<double>& buffer_n,
            test_result& res);
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(HyperParamStats*);
  void update_hyperparam(CCPFHyperParamStats*);

 protected:
  HPFCouplingInterface* rmdl;
  HPFInterface* smdl;
};

class CCPFNormal : public Model {
 public:
  CCPFNormal();
  void initialize_with_sparse_matrix(SparseMatrix& sm, int n_component_, int n_trunc_,
                                     double xi_, double tau_);
  void serialize(string&);
  void deserialize(string&);
  int get_n_trunc();
  int get_n_component();
  double get_sigma2();
  HPFNormalCouplingInterface* get_rmdl();
  HPFNormalInterface* get_smdl();
  void set_rmdl(HPFNormalCouplingInterface* rmdl_);
  void set_smdl(HPFNormalInterface* smdl_);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              HyperParamStats* stats);
  void update(unsigned long int ui, unsigned long int ii, double val,
              vector<double>& buffer_k, vector<double>& buffer_n,
              CCPFHyperParamStats* stats);
  void test(unsigned long int ui, unsigned long int ii, double val,
            vector<double>& buffer_k, vector<double>& buffer_n,
            test_result& res);
  double predict(unsigned long int ui, unsigned long int ii,
                 vector<double>& buffer_k);
  HyperParamStats* create_hyperparam_stats();
  void update_hyperparam(HyperParamStats*);
  void update_hyperparam(CCPFHyperParamStats*);

 protected:
  HPFNormalCouplingInterface* rmdl;
  HPFNormalInterface* smdl;
};

#endif  // CCPF_H
