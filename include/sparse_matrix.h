#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H
#include "utils.h"
#include <boost/fusion/adapted.hpp>
using namespace std;

BOOST_FUSION_ADAPT_STRUCT(entry,
                          (unsigned long int, user)(unsigned long int,
                                                    item)(double, response))
BOOST_FUSION_ADAPT_STRUCT(missing_entry,
                          (unsigned long int, user)(unsigned long int, item))

class SparseMatrix {
 public:
  SparseMatrix(const char*, const char*, const char*, const char*, const char*,
               const char*, const char*, double, double);
  const itemmaps& get_train();
  const itemmaps& get_val();
  const itemmaps& get_test();
  const itemmaps& get_valzero();
  const itemmaps& get_testzero();
  const itemmap& get_user2ind();
  const itemmap& get_item2ind();
  const vector<unsigned long int>& get_ind2user();
  const vector<unsigned long int>& get_ind2item();
  const triplets& get_ind2train();
  const triplets& get_ind2val();
  const triplets& get_ind2test();
  const tuples& get_ind2valzero();
  const tuples& get_ind2testzero();
  unsigned long int get_n_user();
  unsigned long int get_n_item();
  unsigned long long int get_n_total();
  unsigned long long int get_n_total_nonzero();
  unsigned long long int get_n_train_nonzero();
  unsigned long long int get_n_val_nonzero();
  unsigned long long int get_n_test_nonzero();
  unsigned long long int get_n_total_zero();
  unsigned long long int get_n_train_zero();
  unsigned long long int get_n_val_zero();
  unsigned long long int get_n_eff_val_zero();
  unsigned long long int get_n_test_zero();
  unsigned long long int get_n_eff_test_zero();
  double get_max_response();
  double get_min_response();
  double get_sparsity();

  bool is_nz_train(unsigned long int, unsigned long int);
  bool is_nz_val(unsigned long int, unsigned long int);
  bool is_nz_test(unsigned long int, unsigned long int);
  bool is_nz(unsigned long int, unsigned long int);
  bool is_z(unsigned long int, unsigned long int);
  bool is_cond_z_train(unsigned long int, unsigned long int);
  bool is_z_train(unsigned long int, unsigned long int);
  bool is_cond_z_val(unsigned long int, unsigned long int);
  bool is_z_val(unsigned long int, unsigned long int);
  bool is_cond_z_test(unsigned long int, unsigned long int);
  bool is_z_test(unsigned long int, unsigned long int);

  void sample_nz(triplets&);
  void sample_z(triplets&);
  void sample(triplets&);
  void sample_nz(entry&);
  void sample_z(entry&);
  void sample(entry&);

  void read_users(const char* users_file);
  void read_items(const char* items_file);
  void read_train(const char* train_file);
  void read_val(const char* val_file);
  void read_test(const char* test_file);
  void read_val_zero(const char* val_zero_file);
  void read_test_zero(const char* test_zero_file);

 private:
  itemmaps train;
  itemmaps val;
  itemmaps test;
  itemmaps valzero;
  itemmaps testzero;
  itemmap user2ind;
  itemmap item2ind;
  triplets ind2train;
  triplets ind2val;
  triplets ind2test;
  tuples ind2valzero;
  tuples ind2testzero;
  vector<unsigned long int> ind2user;
  vector<unsigned long int> ind2item;
  unsigned long int n_user;
  unsigned long int n_item;
  unsigned long long int n_total;
  unsigned long long int n_total_nonzero;
  unsigned long long int n_train_nonzero;
  unsigned long long int n_val_nonzero;
  unsigned long long int n_test_nonzero;
  unsigned long long int n_total_zero;
  unsigned long long int n_train_zero;
  unsigned long long int n_val_zero;
  unsigned long long int n_eff_val_zero;
  unsigned long long int n_test_zero;
  unsigned long long int n_eff_test_zero;
  double validation_ratio;
  double test_ratio;
  double max_response;
  double min_response;
  double sparsity;
};

#endif  // SPARSE_MATRIX_H
