#include "utils.h"
#include "sparse_matrix.h"
#include <boost/spirit/include/qi.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

using namespace std;
namespace bqi = boost::spirit::qi;
namespace bio = boost::iostreams;

SparseMatrix::SparseMatrix(const char* users_file, const char* items_file,
                           const char* train_file, const char* val_file,
                           const char* test_file, const char* val_zero_file,
                           const char* test_zero_file,
                           double validation_ratio_ = 0.1,
                           double test_ratio_ = 0.20) {
  validation_ratio = validation_ratio_;
  test_ratio = test_ratio_;
  max_response = numeric_limits<double>::min();
  min_response = numeric_limits<double>::max();
  this->read_users(users_file);
  this->read_items(items_file);
  this->read_train(train_file);
  this->read_val(val_file);
  this->read_test(test_file);
  this->read_val_zero(val_zero_file);
  this->read_test_zero(test_zero_file);
  n_total_nonzero = n_train_nonzero + n_test_nonzero + n_val_nonzero;
  n_total = n_user * n_item;
  n_total_zero = n_total - n_total_nonzero;
  n_val_zero = n_total_zero * validation_ratio;
  n_test_zero = n_total_zero * test_ratio;
  n_train_zero = n_total_zero - n_val_zero - n_test_zero;
  sparsity = 1.0 - (double)n_total_nonzero / (double)n_total;
}

const itemmaps& SparseMatrix::get_train() { return train; };
const itemmaps& SparseMatrix::get_val() { return val; };
const itemmaps& SparseMatrix::get_test() { return test; };
const itemmaps& SparseMatrix::get_valzero() { return valzero; };
const itemmaps& SparseMatrix::get_testzero() { return testzero; };
const itemmap& SparseMatrix::get_user2ind() { return user2ind; };
const itemmap& SparseMatrix::get_item2ind() { return item2ind; };
const vector<unsigned long int>& SparseMatrix::get_ind2user() {
  return ind2user;
};
const vector<unsigned long int>& SparseMatrix::get_ind2item() {
  return ind2item;
};
const triplets& SparseMatrix::get_ind2train() { return ind2train; };
const triplets& SparseMatrix::get_ind2val() { return ind2val; };
const triplets& SparseMatrix::get_ind2test() { return ind2test; };
const tuples& SparseMatrix::get_ind2valzero() { return ind2valzero; };
const tuples& SparseMatrix::get_ind2testzero() { return ind2testzero; };
unsigned long int SparseMatrix::get_n_user() { return n_user; };
unsigned long int SparseMatrix::get_n_item() { return n_item; };
unsigned long long int SparseMatrix::get_n_total() { return n_total; };
unsigned long long int SparseMatrix::get_n_total_nonzero() {
  return n_total_nonzero;
};
unsigned long long int SparseMatrix::get_n_train_nonzero() {
  return n_train_nonzero;
};
unsigned long long int SparseMatrix::get_n_val_nonzero() {
  return n_val_nonzero;
};
unsigned long long int SparseMatrix::get_n_test_nonzero() {
  return n_test_nonzero;
};
unsigned long long int SparseMatrix::get_n_total_zero() {
  return n_total_zero;
};
unsigned long long int SparseMatrix::get_n_train_zero() {
  return n_train_zero;
};
unsigned long long int SparseMatrix::get_n_val_zero() { return n_val_zero; };
unsigned long long int SparseMatrix::get_n_eff_val_zero() {
  return n_eff_val_zero;
};
unsigned long long int SparseMatrix::get_n_test_zero() { return n_test_zero; };
unsigned long long int SparseMatrix::get_n_eff_test_zero() {
  return n_eff_test_zero;
};
double SparseMatrix::get_max_response() { return max_response; };
double SparseMatrix::get_min_response() { return min_response; };
double SparseMatrix::get_sparsity() { return sparsity; };

void SparseMatrix::read_users(const char* users_file) {
  bio::mapped_file mmap(users_file, bio::mapped_file::readonly);
  auto f = mmap.const_data();
  auto l = f + mmap.size();
  bool ok = bqi::phrase_parse(f, l, bqi::int_ % bqi::eol, bqi::blank, ind2user);
  if (ok)
    std::cout << "Users file : parse success\n";
  else
    std::cerr << "Users file : parse failed: '" << std::string(f, l) << "'\n";

  if (f != l)
    std::cerr << "Users file : trailing unparsed: '" << std::string(f, l)
              << "'\n";

  n_user = ind2user.size();
  train.resize(n_user);
  test.resize(n_user);
  val.resize(n_user);
  valzero.resize(n_user);
  testzero.resize(n_user);
  for (size_t i = 0; i < n_user; ++i) user2ind[ind2user[i]] = i;
}

void SparseMatrix::read_items(const char* items_file) {
  bio::mapped_file mmap(items_file, bio::mapped_file::readonly);
  auto f = mmap.const_data();
  auto l = f + mmap.size();
  bool ok = bqi::phrase_parse(f, l, bqi::int_ % bqi::eol, bqi::blank, ind2item);
  if (ok)
    std::cout << "Items file : parse success\n";
  else
    std::cerr << "Items file : parse failed: '" << std::string(f, l) << "'\n";

  if (f != l)
    std::cerr << "Items file : trailing unparsed: '" << std::string(f, l)
              << "'\n";

  n_item = ind2item.size();
  for (size_t i = 0; i < n_item; ++i) item2ind[ind2item[i]] = i;
}

void SparseMatrix::read_train(const char* train_file) {
  bio::mapped_file mmap(train_file, bio::mapped_file::readonly);
  auto f = mmap.const_data();
  auto l = f + mmap.size();
  bool ok =
      bqi::phrase_parse(f, l, (bqi::int_ > bqi::int_ > bqi::double_) % bqi::eol,
                        bqi::blank, ind2train);
  if (ok)
    std::cout << "Train file : parse success\n";
  else
    std::cerr << "Train file : parse failed: '" << std::string(f, l) << "'\n";

  if (f != l)
    std::cerr << "Train file : trailing unparsed: '" << std::string(f, l)
              << "'\n";

  n_train_nonzero = ind2train.size();

  unsigned long int ui, ii;
  double response;
  for (size_t i = 0; i < n_train_nonzero; ++i) {
    response = ind2train[i].response;
    ui = user2ind[ind2train[i].user];
    ii = item2ind[ind2train[i].item];
    train[ui][ii] = response;
    ind2train[i].user = ui;
    ind2train[i].item = ii;
    if (response > max_response) max_response = response;
    if (response < min_response) min_response = response;
  }
}

void SparseMatrix::read_val(const char* val_file) {
  bio::mapped_file mmap(val_file, bio::mapped_file::readonly);
  auto f = mmap.const_data();
  auto l = f + mmap.size();
  bool ok =
      bqi::phrase_parse(f, l, (bqi::int_ > bqi::int_ > bqi::double_) % bqi::eol,
                        bqi::blank, ind2val);
  if (ok)
    std::cout << "Validation file : parse success\n";
  else
    std::cerr << "Validation file : parse failed: '" << std::string(f, l)
              << "'\n";

  if (f != l)
    std::cerr << "Validation file : trailing unparsed: '" << std::string(f, l)
              << "'\n";

  n_val_nonzero = ind2val.size();
  unsigned long int ui, ii;
  double response;
  for (size_t i = 0; i < n_val_nonzero; ++i) {
    response = ind2val[i].response;
    ui = user2ind[ind2val[i].user];
    ii = item2ind[ind2val[i].item];
    val[ui][ii] = response;
    ind2val[i].user = ui;
    ind2val[i].item = ii;
    if (response > max_response) max_response = response;
    if (response < min_response) min_response = response;
  }
}

void SparseMatrix::read_test(const char* test_file) {
  bio::mapped_file mmap(test_file, bio::mapped_file::readonly);
  auto f = mmap.const_data();
  auto l = f + mmap.size();
  bool ok =
      bqi::phrase_parse(f, l, (bqi::int_ > bqi::int_ > bqi::double_) % bqi::eol,
                        bqi::blank, ind2test);
  if (ok)
    std::cout << "Test file : parse success\n";
  else
    std::cerr << "Test file : parse failed: '" << std::string(f, l) << "'\n";

  if (f != l)
    std::cerr << "Test file : trailing unparsed: '" << std::string(f, l)
              << "'\n";

  n_test_nonzero = ind2test.size();
  unsigned long int ui, ii;
  double response;
  for (size_t i = 0; i < n_test_nonzero; ++i) {
    response = ind2test[i].response;
    ui = user2ind[ind2test[i].user];
    ii = item2ind[ind2test[i].item];
    test[ui][ii] = response;
    ind2test[i].user = ui;
    ind2test[i].item = ii;
    if (response > max_response) max_response = response;
    if (response < min_response) min_response = response;
  }
}

void SparseMatrix::read_val_zero(const char* val_zero_file) {
  bio::mapped_file mmap(val_zero_file, bio::mapped_file::readonly);
  auto f = mmap.const_data();
  auto l = f + mmap.size();
  bool ok = bqi::phrase_parse(f, l, (bqi::int_ > bqi::int_) % bqi::eol,
                              bqi::blank, ind2valzero);
  if (ok)
    std::cout << "Validation Zero file : parse success\n";
  else
    std::cerr << "Validation Zero file : parse failed: '" << std::string(f, l)
              << "'\n";

  if (f != l)
    std::cerr << "Validation Zero file : trailing unparsed: '"
              << std::string(f, l) << "'\n";

  n_eff_val_zero = ind2valzero.size();
  unsigned long int ui, ii;
  for (size_t i = 0; i < n_eff_val_zero; ++i) {
    ui = user2ind[ind2valzero[i].user];
    ii = item2ind[ind2valzero[i].item];
    valzero[ui][ii] = 0;
    ind2valzero[i].user = ui;
    ind2valzero[i].item = ii;
  }
}

void SparseMatrix::read_test_zero(const char* test_zero_file) {
  bio::mapped_file mmap(test_zero_file, bio::mapped_file::readonly);
  auto f = mmap.const_data();
  auto l = f + mmap.size();
  bool ok = bqi::phrase_parse(f, l, (bqi::int_ > bqi::int_) % bqi::eol,
                              bqi::blank, ind2testzero);
  if (ok)
    std::cout << "Test Zero file : parse success\n";
  else
    std::cerr << "Test Zero file : parse failed: '" << std::string(f, l)
              << "'\n";

  if (f != l)
    std::cerr << "Test Zero file : trailing unparsed: '" << std::string(f, l)
              << "'\n";

  n_eff_test_zero = ind2testzero.size();
  unsigned long int ui, ii;
  for (size_t i = 0; i < n_eff_test_zero; ++i) {
    ui = user2ind[ind2testzero[i].user];
    ii = item2ind[ind2testzero[i].item];
    testzero[ui][ii] = 0;
    ind2testzero[i].user = ui;
    ind2testzero[i].item = ii;
  }
}

bool SparseMatrix::is_nz_train(unsigned long int ui, unsigned long int ii) {
  return ((!train[ui].empty()) && (train[ui].count(ii)));
}

bool SparseMatrix::is_nz_val(unsigned long int ui, unsigned long int ii) {
  return ((!val[ui].empty()) && (val[ui].count(ii)));
}

bool SparseMatrix::is_nz_test(unsigned long int ui, unsigned long int ii) {
  return ((!test[ui].empty()) && (test[ui].count(ii)));
}

bool SparseMatrix::is_nz(unsigned long int ui, unsigned long int ii) {
  return ((this->is_nz_train(ui, ii)) || (this->is_nz_val(ui, ii)) ||
          (this->is_nz_test(ui, ii)));
}

bool SparseMatrix::is_z(unsigned long int ui, unsigned long int ii) {
  return (!this->is_nz(ui, ii));
}

bool SparseMatrix::is_cond_z_train(unsigned long int ui, unsigned long int ii) {
  return ((!this->is_cond_z_val(ui, ii)) && (!this->is_cond_z_test(ui, ii)));
}

bool SparseMatrix::is_z_train(unsigned long int ui, unsigned long int ii) {
  return ((this->is_z(ui, ii)) && (this->is_cond_z_train(ui, ii)));
}

bool SparseMatrix::is_cond_z_val(unsigned long int ui, unsigned long int ii) {
  return ((!valzero[ui].empty()) && (valzero[ui].count(ii)));
}

bool SparseMatrix::is_z_val(unsigned long int ui, unsigned long int ii) {
  return ((this->is_z(ui, ii)) && (this->is_cond_z_val(ui, ii)));
}

bool SparseMatrix::is_cond_z_test(unsigned long int ui, unsigned long int ii) {
  return ((!testzero[ui].empty()) && (testzero[ui].count(ii)));
}

bool SparseMatrix::is_z_test(unsigned long int ui, unsigned long int ii) {
  return ((this->is_z(ui, ii)) && (this->is_cond_z_test(ui, ii)));
}

void SparseMatrix::sample_nz(triplets& batch) {
  unsigned long int batchsize = batch.size();
  unsigned long int ri, ui, ii;
  double response;
  for (size_t i = 0; i < batchsize; ++i) {
    ri = rand() % n_train_nonzero;
    ui = ind2train[ri].user;
    ii = ind2train[ri].item;
    response = train[ui][ii];
    batch[i].user = ui;
    batch[i].item = ii;
    batch[i].response = response;
  }
};

void SparseMatrix::sample_nz(entry& e) {
  unsigned long int ri = rand() % n_train_nonzero;
  unsigned long int ui = ind2train[ri].user;
  unsigned long int ii = ind2train[ri].item;
  e.user = ui;
  e.item = ii;
  e.response = train[ui][ii];
};

void SparseMatrix::sample_z(triplets& batch) {
  unsigned long int batchsize = batch.size();
  unsigned long int ui, ii;
  unsigned long int i = 0;
  while (i < batchsize) {
    ui = rand() % n_user;
    ii = rand() % n_item;
    if (this->is_z_train(ui, ii)) {
      batch[i].user = ui;
      batch[i].item = ii;
      batch[i].response = 0;
      i += 1;
    }
  }
};

void SparseMatrix::sample_z(entry& e) {
  unsigned long int ui, ii;
  bool invalid = 1;
  while (invalid) {
    ui = rand() % n_user;
    ii = rand() % n_item;
    if (this->is_z_train(ui, ii)) {
      e.user = ui;
      e.item = ii;
      e.response = 0;
      invalid = 0;
    }
  }
};

void SparseMatrix::sample(triplets& batch) {
  unsigned long int batchsize = batch.size();
  unsigned long int ui, ii;
  unsigned long int i = 0;
  while (i < batchsize) {
    ui = rand() % n_user;
    ii = rand() % n_item;
    if (this->is_nz_train(ui, ii)) {
      batch[i].user = ui;
      batch[i].item = ii;
      batch[i].response = train[ui][ii];
      i += 1;
    } else if (this->is_z_train(ui, ii)) {
      batch[i].user = ui;
      batch[i].item = ii;
      batch[i].response = 0;
      i += 1;
    }
  }
};

void SparseMatrix::sample(entry& e) {
  unsigned long int ui, ii;
  bool invalid = 1;
  while (invalid) {
    ui = rand() % n_user;
    ii = rand() % n_item;
    if (is_nz_train(ui, ii)) {
      e.user = ui;
      e.item = ii;
      e.response = train[ui][ii];
      invalid = 0;
    } else if (is_z_train(ui, ii)) {
      e.user = ui;
      e.item = ii;
      e.response = 0;
      invalid = 0;
    }
  }
};
