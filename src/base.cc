#include "utils.h"
#include "base.h"
#include <omp.h>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>

using namespace std;
namespace bfs = boost::filesystem;
namespace bptree = boost::property_tree;

Configuration::Configuration(){};

unsigned long int Configuration::get_conf_num() { return conf_num; };

string& Configuration::get_in_dir() { return in_dir; };

string& Configuration::get_out_dir() { return out_dir; };

string& Configuration::get_response_model() { return response_model; };

string& Configuration::get_sparsity_model() { return sparsity_model; };

string& Configuration::get_dataset() { return dataset; };

string& Configuration::get_sample_method() { return sample_method; };

unsigned long int Configuration::get_batchsize() { return batchsize; };

unsigned long int Configuration::get_test_every() { return test_every; };

unsigned long int Configuration::get_save_pred_every() {
  return save_pred_every;
};

unsigned long int Configuration::get_save_every() { return save_every; };

unsigned long int Configuration::get_max_iter() { return max_iter; };

int Configuration::get_n_component() { return n_component; };

double Configuration::get_xi() { return xi; };

double Configuration::get_tau() { return tau; };

double Configuration::get_test_ratio() { return test_ratio; };

double Configuration::get_validation_ratio() { return validation_ratio; };

unsigned long int Configuration::get_max_save_iter() {
  return max_iter / test_every;
};

void Configuration::read(string& fname, unsigned long int conf_num_) {
  // Read configuration from ini file
  conf_num = conf_num_;
  bptree::ptree pt;
  bptree::ini_parser::read_ini(fname, pt);
  string strconfnum = to_string(static_cast<long long>(conf_num));
  in_dir = pt.get<string>(strconfnum + ".in_dir");
  out_dir = pt.get<string>(strconfnum + ".out_dir");
  out_dir += "exp_" + strconfnum + "/";
  response_model = pt.get<string>(strconfnum + ".response_model");
  sparsity_model = pt.get<string>(strconfnum + ".sparsity_model");
  dataset = pt.get<string>(strconfnum + ".dataset");
  sample_method = pt.get<string>(strconfnum + ".sample_method");
  batchsize = pt.get<unsigned long int>(strconfnum + ".batchsize");
  test_every = pt.get<unsigned long int>(strconfnum + ".test_every");
  save_pred_every = pt.get<unsigned long int>(strconfnum + ".save_pred_every");
  save_every = pt.get<unsigned long int>(strconfnum + ".save_every");
  max_iter = pt.get<unsigned long int>(strconfnum + ".max_iter");
  n_component = pt.get<int>(strconfnum + ".n_component");
  xi = pt.get<double>(strconfnum + ".xi");
  tau = pt.get<double>(strconfnum + ".tau");
  validation_ratio = pt.get<double>(strconfnum + ".validation_ratio");
  test_ratio = pt.get<double>(strconfnum + ".test_ratio");
}

void Configuration::write_json(bptree::ptree& pt) {
  // Append configuration to boost property_tree
  pt.put("conf_num", conf_num);
  pt.put("in_dir", in_dir);
  pt.put("out_dir", out_dir);
  pt.put("response_model", response_model);
  pt.put("sparsity_model", sparsity_model);
  pt.put("dataset", dataset);
  pt.put("sample_method", sample_method);
  pt.put("batchsize", batchsize);
  pt.put("test_every", test_every);
  pt.put("save_pred_every", save_pred_every);
  pt.put("save_every", save_every);
  pt.put("max_iter", max_iter);
  pt.put("n_component", n_component);
  pt.put("xi", xi);
  pt.put("tau", tau);
  pt.put("test_ratio", test_ratio);
  pt.put("validation_ratio", validation_ratio);
}

Result::Result(Configuration& cfg_, SparseMatrix& sm, int n_trunc_)
    : cfg(cfg_) {
  n_trunc = n_trunc_;
  n_user = sm.get_n_user();
  n_item = sm.get_n_item();
  n_total_nonzero = sm.get_n_total_nonzero();
  n_train_nonzero = sm.get_n_train_nonzero();
  n_test_nonzero = sm.get_n_test_nonzero();
  n_val_nonzero = sm.get_n_val_nonzero();
  n_total_zero = sm.get_n_total_zero();
  n_train_zero = sm.get_n_train_zero();
  n_test_zero = sm.get_n_test_zero();
  n_val_zero = sm.get_n_val_zero();
  n_eff_test_zero = sm.get_n_eff_test_zero();
  n_eff_val_zero = sm.get_n_eff_val_zero();
  sparsity = sm.get_sparsity();
  max_response = sm.get_max_response();
  min_response = sm.get_min_response();

  max_save_iter = cfg.get_max_save_iter();
  val_z_tll.resize(max_save_iter);
  val_nz_tll.resize(max_save_iter);
  val_cond_nz_tll.resize(max_save_iter);
  val_tll.resize(max_save_iter);
  z_tll.resize(max_save_iter);
  nz_tll.resize(max_save_iter);
  cond_nz_tll.resize(max_save_iter);
  tll.resize(max_save_iter);
  val_z_err.resize(max_save_iter);
  val_nz_err.resize(max_save_iter);
  val_cond_nz_err.resize(max_save_iter);
  val_err.resize(max_save_iter);
  z_err.resize(max_save_iter);
  nz_err.resize(max_save_iter);
  cond_nz_err.resize(max_save_iter);
  err.resize(max_save_iter);
  recall20.resize(max_save_iter);
  recall50.resize(max_save_iter);
  recall100.resize(max_save_iter);
  map20.resize(max_save_iter);
  map50.resize(max_save_iter);
  map100.resize(max_save_iter);
  ndcg20.resize(max_save_iter);
  ndcg50.resize(max_save_iter);
  ndcg100.resize(max_save_iter);
  auc.resize(max_save_iter);
  index = 0;
}

void Result::update(double val_z_tll_sum, double val_nz_tll_sum,
                    double val_cond_nz_tll_sum, double val_z_err_sum,
                    double val_nz_err_sum, double val_cond_nz_err_sum,
                    double z_tll_sum, double nz_tll_sum, double cond_nz_tll_sum,
                    double z_err_sum, double nz_err_sum, double cond_nz_err_sum,
                    double recall20_sum, double recall50_sum,
                    double recall100_sum, double map20_sum, double map50_sum,
                    double map100_sum, double ndcg20_sum, double ndcg50_sum,
                    double ndcg100_sum, double auc_,
                    unsigned long long int n_eff_user) {
  // Adjust sum statistics for the missing entries
  val_z_tll_sum = val_z_tll_sum / n_eff_val_zero * n_val_zero;
  val_z_err_sum = val_z_err_sum / n_eff_val_zero * n_val_zero;
  z_tll_sum = z_tll_sum / n_eff_test_zero * n_test_zero;
  z_err_sum = z_err_sum / n_eff_test_zero * n_test_zero;

  // Normalize statistics for the validation set
  val_z_tll[index] = val_z_tll_sum / n_val_zero;
  val_nz_tll[index] = val_nz_tll_sum / n_val_nonzero;
  val_cond_nz_tll[index] = val_cond_nz_tll_sum / n_val_nonzero;
  val_tll[index] =
      (val_nz_tll_sum + val_z_tll_sum) / (n_val_nonzero + n_val_zero);
  val_z_err[index] = sqrt(val_z_err_sum / n_val_zero);
  val_nz_err[index] = sqrt(val_nz_err_sum / n_val_nonzero);
  val_cond_nz_err[index] = sqrt(val_cond_nz_err_sum / n_val_nonzero);
  val_err[index] =
      sqrt((val_nz_err_sum + val_z_err_sum) / (n_val_nonzero + n_val_zero));

  // Normalize statistics for the test set
  z_tll[index] = z_tll_sum / n_test_zero;
  nz_tll[index] = nz_tll_sum / n_test_nonzero;
  cond_nz_tll[index] = cond_nz_tll_sum / n_test_nonzero;
  tll[index] = (nz_tll_sum + z_tll_sum) / (n_test_nonzero + n_test_zero);
  z_err[index] = sqrt(z_err_sum / n_test_zero);
  nz_err[index] = sqrt(nz_err_sum / n_test_nonzero);
  cond_nz_err[index] = sqrt(cond_nz_err_sum / n_test_nonzero);
  err[index] = sqrt((nz_err_sum + z_err_sum) / (n_test_nonzero + n_test_zero));

  // Normalize ranking statistics
  recall20[index] = recall20_sum / n_eff_user;
  recall50[index] = recall50_sum / n_eff_user;
  recall100[index] = recall100_sum / n_eff_user;
  map20[index] = map20_sum / n_eff_user;
  map50[index] = map50_sum / n_eff_user;
  map100[index] = map100_sum / n_eff_user;
  ndcg20[index] = ndcg20_sum / n_eff_user;
  ndcg50[index] = ndcg50_sum / n_eff_user;
  ndcg100[index] = ndcg100_sum / n_eff_user;
  auc[index] = auc_;
  cout << z_tll[index] << " : " << nz_tll[index] << " : " << cond_nz_tll[index]
       << " : " << tll[index] << " : ";
  cout << z_err[index] << " : " << nz_err[index] << " : " << cond_nz_err[index]
       << " : " << err[index] << " : " << auc[index] << endl;
  cout << recall20[index] << " : " << recall50[index] << " : "
       << recall100[index] << " : " << map20[index] << " : " << map50[index]
       << " : " << map100[index] << " : " << ndcg20[index] << " : "
       << ndcg50[index] << " : " << ndcg100[index] << endl;
  index += 1;
}

void Result::write_json(bptree::ptree& pt) {
  // Append configuration to boost property_tree
  pt.put("n_trunc", n_trunc);
  pt.put("n_user", n_user);
  pt.put("n_item", n_item);
  pt.put("n_total_nonzero", n_total_nonzero);
  pt.put("n_train_nonzero", n_train_nonzero);
  pt.put("n_test_nonzero", n_test_nonzero);
  pt.put("n_val_nonzero", n_val_nonzero);
  pt.put("n_total_zero", n_total_zero);
  pt.put("n_train_zero", n_train_zero);
  pt.put("n_val_zero", n_val_zero);
  pt.put("n_test_zero", n_test_zero);
  pt.put("n_eff_val_zero", n_eff_val_zero);
  pt.put("n_eff_test_zero", n_eff_test_zero);
  pt.put("sparsity", sparsity);
  pt.put("max_response", max_response);
  pt.put("min_response", min_response);

  string key;
  key = "val_z_tll";
  add_vector_to_ptree(pt, key, val_z_tll, index);
  key = "val_nz_tll";
  add_vector_to_ptree(pt, key, val_nz_tll, index);
  key = "val_cond_nz_tll";
  add_vector_to_ptree(pt, key, val_cond_nz_tll, index);
  key = "val_tll";
  add_vector_to_ptree(pt, key, val_tll, index);
  key = "val_z_err";
  add_vector_to_ptree(pt, key, val_z_err, index);
  key = "val_nz_err";
  add_vector_to_ptree(pt, key, val_nz_err, index);
  key = "val_cond_nz_err";
  add_vector_to_ptree(pt, key, val_cond_nz_err, index);
  key = "val_err";
  add_vector_to_ptree(pt, key, val_err, index);

  key = "z_tll";
  add_vector_to_ptree(pt, key, z_tll, index);
  key = "nz_tll";
  add_vector_to_ptree(pt, key, nz_tll, index);
  key = "cond_nz_tll";
  add_vector_to_ptree(pt, key, cond_nz_tll, index);
  key = "tll";
  add_vector_to_ptree(pt, key, tll, index);
  key = "z_err";
  add_vector_to_ptree(pt, key, z_err, index);
  key = "nz_err";
  add_vector_to_ptree(pt, key, nz_err, index);
  key = "cond_nz_err";
  add_vector_to_ptree(pt, key, cond_nz_err, index);
  key = "err";
  add_vector_to_ptree(pt, key, err, index);

  key = "recall20";
  add_vector_to_ptree(pt, key, recall20, index);
  key = "recall50";
  add_vector_to_ptree(pt, key, recall50, index);
  key = "recall100";
  add_vector_to_ptree(pt, key, recall100, index);
  key = "map20";
  add_vector_to_ptree(pt, key, map20, index);
  key = "map50";
  add_vector_to_ptree(pt, key, map50, index);
  key = "map100";
  add_vector_to_ptree(pt, key, map100, index);
  key = "ndcg20";
  add_vector_to_ptree(pt, key, ndcg20, index);
  key = "ndcg50";
  add_vector_to_ptree(pt, key, ndcg50, index);
  key = "ndcg100";
  add_vector_to_ptree(pt, key, ndcg100, index);

  key = "auc";
  add_vector_to_ptree(pt, key, auc, index);
}

void Result::save() {
  // Save to json file
  bptree::ptree pt;
  cfg.write_json(pt);
  write_json(pt);
  string out_dir = cfg.get_out_dir();
  bfs::path out_dir_fp(out_dir);
  if (!bfs::exists(out_dir_fp)) bfs::create_directories(out_dir_fp);
  string fname = out_dir + "result.json";
  bptree::json_parser::write_json(fname, pt);
}

void HyperParamStats::clean(){}

void Model::fit(SparseMatrix& sm, unsigned long int batchsize,
                unsigned long int max_iter, unsigned long int test_every,
                unsigned long int save_every, unsigned long int save_pred_every,
                string& sample_method, string& out_dir, Result& result,
                unsigned long long int n_user_max = 1000) {
  unsigned long long int n_val_nonzero = sm.get_n_val_nonzero();
  unsigned long long int n_test_nonzero = sm.get_n_test_nonzero();
  unsigned long long int n_val_zero = sm.get_n_val_zero();
  unsigned long long int n_eff_val_zero = sm.get_n_eff_val_zero();
  unsigned long long int n_test_zero = sm.get_n_test_zero();
  unsigned long long int n_eff_test_zero = sm.get_n_eff_test_zero();
  unsigned long long int n_user = sm.get_n_user();
  unsigned long long int n_item = sm.get_n_item();
  unsigned long long int n_user_ranking = min(n_user_max, n_user);
  int n_component = get_n_component();
  int n_trunc = get_n_trunc();
  const triplets& ind2val = sm.get_ind2val();
  const triplets& ind2test = sm.get_ind2test();
  const tuples& ind2valzero = sm.get_ind2valzero();
  const tuples& ind2testzero = sm.get_ind2testzero();

  double n_eff_user = 0;
  int k_max = 100;
  vector<double> weight(k_max);
  for (size_t i = 0; i < k_max; ++i) weight[i] = 1.0 / log2(i + 2);
  const itemmaps& test_map = sm.get_test();
  const itemmaps& train_map = sm.get_train();
  const itemmaps& validation_map = sm.get_val();

  predictions preds;
  preds.resize(n_test_nonzero + n_eff_test_zero);

  double val_z_tll_sum, val_nz_tll_sum, val_cond_nz_tll_sum, val_z_err_sum,
      val_nz_err_sum, val_cond_nz_err_sum, z_tll_sum, nz_tll_sum,
      cond_nz_tll_sum, z_err_sum, nz_err_sum, cond_nz_err_sum, recall20_sum,
      recall50_sum, recall100_sum, map20_sum, map50_sum, map100_sum, ndcg20_sum,
      ndcg50_sum, ndcg100_sum;
  double auc;
  for (size_t iter = 1; iter < max_iter + 1; ++iter) {
#pragma omp parallel shared(preds, n_item, n_user, n_component, n_trunc,    \
                            k_max, weight, test_map, train_map,             \
                            validation_map, val_z_tll_sum, val_nz_tll_sum,  \
                            val_cond_nz_tll_sum, val_z_err_sum,             \
                            val_nz_err_sum, val_cond_nz_err_sum, z_tll_sum, \
                            nz_tll_sum, cond_nz_tll_sum, z_err_sum,         \
                            nz_err_sum, cond_nz_err_sum)
    {
      const int thread_id = omp_get_thread_num();
      const int nthreads = omp_get_num_threads();
      srand(int(time(NULL)) ^ thread_id);
      entry e;
      HyperParamStats* stats = create_hyperparam_stats();
      test_result res;
      vector<double> buffer_n(n_trunc, 0);
      vector<double> buffer_k(n_component, 0);
      vector<double> hit(k_max, 0);
      vector<double> recallatk(k_max, 0);
      vector<double> precisionatk(k_max, 0);
      vector<double> avprecisionatk(k_max, 0);
      vector<double> ndcgatk(k_max, 0);
      itemmap test_items;
      itemmap train_items;
      itemmap val_items;
      predictions rankingp;
      rankingp.resize(n_item);
      int n_test_item;

      if (sample_method.compare("nonmissing") == 0) {
#pragma omp for
        for (size_t j = 0; j < batchsize; ++j) {
          sm.sample_nz(e);
          update(e.user, e.item, e.response, buffer_k, buffer_n, stats);
        }
      } else if (sample_method.compare("binary") == 0) {
#pragma omp for
        for (size_t j = 0; j < batchsize; ++j) {
          sm.sample(e);
          if (e.response != 0) e.response = 1;
          update(e.user, e.item, e.response, buffer_k, buffer_n, stats);
        }
      } else if (sample_method.compare("full") == 0) {
#pragma omp for
        for (size_t j = 0; j < batchsize; ++j) {
          sm.sample(e);
          update(e.user, e.item, e.response, buffer_k, buffer_n, stats);
        }
      } else
        cerr << "Error: Unknown sample method\n";

#pragma omp critical
      { update_hyperparam(stats); }
      if (stats)
        stats->clean();
      delete stats;

      if (iter % test_every == 0) {
        val_z_tll_sum = val_nz_tll_sum = val_cond_nz_tll_sum = 0;
        val_z_err_sum = val_nz_err_sum = val_cond_nz_err_sum = 0;
        z_tll_sum = nz_tll_sum = cond_nz_tll_sum = 0;
        z_err_sum = nz_err_sum = cond_nz_err_sum = 0;
        recall20_sum = recall50_sum = recall100_sum = 0;
        map20_sum = map50_sum = map100_sum = 0;
        ndcg20_sum = ndcg50_sum = ndcg100_sum = 0;
        n_eff_user = 0;
        if (sample_method.compare("binary") == 0) {
#pragma omp for reduction(+ : val_nz_tll_sum, val_cond_nz_tll_sum, \
                          val_nz_err_sum, val_cond_nz_err_sum)
          for (size_t i = 0; i < n_val_nonzero; ++i) {
            test(ind2val[i].user, ind2val[i].item, 1.0, buffer_k, buffer_n,
                 res);
            val_nz_tll_sum += res.test_loglik;
            val_cond_nz_tll_sum += res.cond_test_loglik;
            val_nz_err_sum += res.test_error;
            val_cond_nz_err_sum += res.cond_test_error;
          }
#pragma omp for reduction(+ : nz_tll_sum, cond_nz_tll_sum, nz_err_sum, \
                          cond_nz_err_sum)
          for (size_t i = 0; i < n_test_nonzero; ++i) {
            test(ind2test[i].user, ind2test[i].item, 1.0, buffer_k, buffer_n,
                 res);
            nz_tll_sum += res.test_loglik;
            cond_nz_tll_sum += res.cond_test_loglik;
            nz_err_sum += res.test_error;
            cond_nz_err_sum += res.cond_test_error;
            preds[i].pred = res.pred;
            preds[i].label = 1;
          }
        } else {
#pragma omp for reduction(+ : val_nz_tll_sum, val_cond_nz_tll_sum, \
                          val_nz_err_sum, val_cond_nz_err_sum)
          for (size_t i = 0; i < n_val_nonzero; ++i) {
            test(ind2val[i].user, ind2val[i].item, ind2val[i].response,
                 buffer_k, buffer_n, res);
            val_nz_tll_sum += res.test_loglik;
            val_cond_nz_tll_sum += res.cond_test_loglik;
            val_nz_err_sum += res.test_error;
            val_cond_nz_err_sum += res.cond_test_error;
          }
#pragma omp for reduction(+ : nz_tll_sum, cond_nz_tll_sum, nz_err_sum, \
                          cond_nz_err_sum)
          for (size_t i = 0; i < n_test_nonzero; ++i) {
            test(ind2test[i].user, ind2test[i].item, ind2test[i].response,
                 buffer_k, buffer_n, res);
            nz_tll_sum += res.test_loglik;
            cond_nz_tll_sum += res.cond_test_loglik;
            nz_err_sum += res.test_error;
            cond_nz_err_sum += res.cond_test_error;
            preds[i].pred = res.pred;
            preds[i].label = 1;
          }
        }
#pragma omp for reduction(+ : val_z_tll_sum, val_z_err_sum)
        for (size_t i = 0; i < n_eff_val_zero; ++i) {
          test(ind2valzero[i].user, ind2valzero[i].item, 0.0, buffer_k,
               buffer_n, res);
          val_z_tll_sum += res.test_loglik;
          val_z_err_sum += res.test_error;
        }
#pragma omp for reduction(+ : z_tll_sum, z_err_sum)
        for (size_t i = 0; i < n_eff_test_zero; ++i) {
          test(ind2testzero[i].user, ind2testzero[i].item, 0.0, buffer_k,
               buffer_n, res);
          z_tll_sum += res.test_loglik;
          z_err_sum += res.test_error;
          preds[n_test_nonzero + i].pred = res.pred;
          preds[n_test_nonzero + i].label = 0;
        }
#pragma omp for reduction(+ : n_eff_user, recall20_sum, recall50_sum,      \
                          recall100_sum, map20_sum, map50_sum, map100_sum, \
                          ndcg20_sum, ndcg50_sum, ndcg100_sum)
        for (int i = 0; i < n_user_ranking; ++i) {
          test_items = test_map[i];
          train_items = train_map[i];
          val_items = validation_map[i];
          n_test_item = test_items.size();
          n_eff_user += 0;
          if (n_test_item > 0) {
            n_eff_user += 1;
            for (int j = 0; j < n_item; ++j) {
              rankingp[j].pred = -predict(i, j, buffer_k);
              rankingp[j].label = 0;
            }
            for (itemmap::iterator it = test_items.begin();
                 it != test_items.end(); it++)
              rankingp[it->first].label = 1;
            for (itemmap::iterator it = train_items.begin();
                 it != train_items.end(); it++)
              rankingp[it->first].label = -1;
            for (itemmap::iterator it = val_items.begin();
                 it != val_items.end(); it++)
              rankingp[it->first].label = -1;
            calc_metrics(weight, rankingp, hit, recallatk, precisionatk,
                         avprecisionatk, ndcgatk, k_max, n_test_item);
            recall20_sum += recallatk[19];
            recall50_sum += recallatk[49];
            recall100_sum += recallatk[99];
            map20_sum += avprecisionatk[19];
            map50_sum += avprecisionatk[49];
            map100_sum += avprecisionatk[99];
            ndcg20_sum += ndcgatk[19];
            ndcg50_sum += ndcgatk[49];
            ndcg100_sum += ndcgatk[99];
          }
        }
#pragma omp barrier
#pragma omp single
        {
          sort(preds.begin(), preds.end());
          auc = calc_auc(preds);
          result.update(val_z_tll_sum, val_nz_tll_sum, val_cond_nz_tll_sum,
                        val_z_err_sum, val_nz_err_sum, val_cond_nz_err_sum,
                        z_tll_sum, nz_tll_sum, cond_nz_tll_sum, z_err_sum,
                        nz_err_sum, cond_nz_err_sum, recall20_sum, recall50_sum,
                        recall100_sum, map20_sum, map50_sum, map100_sum,
                        ndcg20_sum, ndcg50_sum, ndcg100_sum, auc, n_eff_user);
          result.save();
          // if (iter % save_pred_every == 0) {
          //   dump_predictions(preds);
          // }
          if (iter % save_every == 0) {
            string fname = out_dir + "out.h5";
            serialize(fname);
          }
        }
      }
    }
  }
}
