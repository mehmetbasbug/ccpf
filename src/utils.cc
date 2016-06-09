#include "utils.h"
#include <boost/math/distributions/negative_binomial.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/tools/minima.hpp>
#include <functional>

using namespace std;
namespace bmath = boost::math;
namespace bptree = boost::property_tree;

double trigamma(double x) {
  double a = 0.0001;
  double b = 5.0;
  double b2 = 0.1666666667;
  double b4 = -0.03333333333;
  double b6 = 0.02380952381;
  double b8 = -0.03333333333;
  double value;
  double y;
  double z;
  z = x;
  if (x <= a) {
    value = 1.0 / x / x;
    return value;
  }
  value = 0.0;
  while (z < b) {
    value = value + 1.0 / z / z;
    z = z + 1.0;
  }
  y = 1.0 / z / z;
  value = value + 0.5 * y + (1.0 + y * (b2 + y * (b4 + y * (b6 + y * b8)))) / z;
  return value;
}

double lambertw(const double z) {
  int i;
  int dbgW = 0;
  const double eps = 4.0e-16, em1 = 0.3678794411714423215955237701614608;
  double p, e, t, w;
  if (dbgW) fprintf(stderr, "LambertW: z=%g\n", z);
  if (z < -em1 || std::isinf(z) || std::isnan(z)) {
    fprintf(stderr, "LambertW: bad argument %g, exiting.\n", z);
    return 1;
  }
  if (0.0 == z) return 0.0;
  if (z < -em1 + 1e-4) {  // series near -em1 in sqrt(q)
    double q = z + em1, r = sqrt(q), q2 = q * q, q3 = q2 * q;
    return -1.0 + 2.331643981597124203363536062168 * r -
           1.812187885639363490240191647568 * q +
           1.936631114492359755363277457668 * r * q -
           2.353551201881614516821543561516 * q2 +
           3.066858901050631912893148922704 * r * q2 -
           4.175335600258177138854984177460 * q3 +
           5.858023729874774148815053846119 * r * q3 -
           8.401032217523977370984161688514 * q3 * q;  // error approx 1e-16
  }
  /* initial approx for iteration... */
  if (z < 1.0) { /* series near 0 */
    p = sqrt(2.0 * (2.7182818284590452353602874713526625 * z + 1.0));
    w = -1.0 +
        p * (1.0 +
             p * (-0.333333333333333333333 + p * 0.152777777777777777777777));
  } else
    w = log(z);              /* asymptotic */
  if (z > 3.0) w -= log(w);  /* useful? */
  for (i = 0; i < 10; i++) { /* Halley iteration */
    e = exp(w);
    t = w * e - z;
    p = w + 1.0;
    t /= e * p - 0.5 * (p + 1.0) * t / p;
    w -= t;
    if (fabs(t) < eps * (1.0 + fabs(w))) return w; /* rel-abs error */
  }
  /* should never get here */
  fprintf(stderr, "LambertW: No convergence at z=%g, exiting.\n", z);
  return 1;
}

double logsumexp(double* nums, size_t N) {
  double max_exp = nums[0], sum = 0.0;
  double out = 0.0;
  size_t i;

  for (i = 1; i < N; ++i)
    if (nums[i] > max_exp) max_exp = nums[i];

  for (i = 0; i < N; ++i)
    if (isnormal(nums[i])) sum += exp(nums[i] - max_exp);
  out = log(sum) + max_exp;
  return out;
}

double logsumexp(vector<double>& nums, size_t N) {
  double max_exp = nums[0], sum = 0.0;
  double out = 0.0;
  size_t i;

  for (i = 1; i < N; ++i)
    if (nums[i] > max_exp) max_exp = nums[i];

  for (i = 0; i < N; ++i)
    sum += exp(nums[i] - max_exp);
  out = log(sum) + max_exp;
  return out;
}

double logsumexp2(double* nums, size_t N) {
  if (N == 1) return nums[0];
  double maxp = nums[0], maxn = nums[1], sump = 0.0, sumn = 0.0;
  double logp, logn;
  for (size_t i = 0; i < N; ++i) {
    if (i % 2 == 0) {
      if (nums[i] > maxp) maxp = nums[i];
      if (!std::isinf(nums[i])) sump += exp(nums[i] - maxp);
    } else {
      if (nums[i] > maxn) maxn = nums[i];
      if (!std::isinf(nums[i])) sumn += exp(nums[i] - maxn);
    }
  }
  logp = log(sump) + maxp;
  logn = log(sumn) + maxn;
  if (logp > logn)
    return logp + log(-bmath::expm1(logn - logp));
  else
    return logn + log(-bmath::expm1(logp - logn));
}

double logsumexp2(vector<double>& nums, size_t N) {
  if (N == 1) return nums[0];
  double maxp = nums[0], maxn = nums[1], sump = 0.0, sumn = 0.0;
  double logp, logn;
  for (size_t i = 0; i < N; ++i) {
    if (i % 2 == 0) {
      if (nums[i] > maxp) maxp = nums[i];
      if (!std::isinf(nums[i])) sump += exp(nums[i] - maxp);
    } else {
      if (nums[i] > maxn) maxn = nums[i];
      if (!std::isinf(nums[i])) sumn += exp(nums[i] - maxn);
    }
  }
  logp = log(sump) + maxp;
  logn = log(sumn) + maxn;
  if (logp > logn)
    return logp + log(-bmath::expm1(logn - logp));
  else
    return logn + log(-bmath::expm1(logp - logn));
}

normal_params mle_normal(const triplets& ind2train,
                         unsigned long int n_sample) {
  double sum = 0;
  double sq_sum = 0;
  double val;
  unsigned long int ri;
  unsigned long int n_train_nonzero = ind2train.size();
  for (size_t i = 0; i < n_sample; ++i) {
    ri = rand() % n_train_nonzero;
    val = ind2train[ri].response;
    sum += val;
    sq_sum += val * val;
  }
  double mu = sum / n_sample;
  double sigma2 = sq_sum / n_sample - mu * mu;
  normal_params out;
  out.mu = mu;
  out.sigma2 = sigma2;
  return out;
}

gamma_params mle_gamma(const triplets& ind2train, unsigned long int n_sample) {
  double sum = 0;
  double log_sum = 0;
  double val;
  unsigned long int ri;
  unsigned long int n_train_nonzero = ind2train.size();
  for (size_t i = 0; i < n_sample; ++i) {
    ri = rand() % n_train_nonzero;
    val = ind2train[ri].response;
    sum += val;
    log_sum += log(val);
  }
  double t0 = sum / n_sample;
  double t1 = log(t0);
  double t2 = log_sum / n_sample;
  double a = 0.5 / (t1 - t2);
  double a_new;
  int MAX_ITER = 100;
  double TOLERANCE = 1e-9;
  for (int i = 0; i < MAX_ITER; ++i) {
    a_new = 1.0 / (1.0 / a +
                   (t2 - t1 + log(a) - bmath::digamma(a)) /
                       (a - a * a * trigamma(a)));
    if (abs(a_new / a - 1) < TOLERANCE) break;
    a = a_new;
  }
  gamma_params out;
  out.shape = a;
  out.rate = a / t0;
  return out;
}

inverse_gaussian_params mle_inverse_gaussian(const triplets& ind2train,
                                             unsigned long int n_sample) {
  double sum = 0;
  double inv_sum = 0;
  double val;
  unsigned long int ri;
  unsigned long int n_train_nonzero = ind2train.size();
  for (size_t i = 0; i < n_sample; ++i) {
    ri = rand() % n_train_nonzero;
    val = ind2train[ri].response;
    sum += val;
    inv_sum += 1.0 / val;
  }
  double mu = sum / n_sample;
  double lambda = 1.0 / (inv_sum / n_sample - 1.0 / mu);
  inverse_gaussian_params out;
  out.mu = mu;
  out.lambda = lambda;
  return out;
}

poisson_params mle_poisson(const triplets& ind2train,
                           unsigned long int n_sample) {
  double sum = 0;
  double val;
  unsigned long int ri;
  unsigned long int n_train_nonzero = ind2train.size();
  for (size_t i = 0; i < n_sample; ++i) {
    ri = rand() % n_train_nonzero;
    val = ind2train[ri].response;
    sum += val;
  }
  double lambda = sum / n_sample;
  poisson_params out;
  out.lambda = lambda;
  return out;
}

double binomial_loglik(double r_float, itemmap& val2count, double mean) {
  double loglik = 0.0;
  int r = round(r_float);
  double p = mean / r;
  unsigned long int y, count;
  bmath::binomial binomial(r, p);
  itemmap::iterator it;
  unsigned long int n_sample = 0;
  for (it = val2count.begin(); it != val2count.end(); it++) {
    y = it->first;
    count = it->second;
    n_sample += count;
    loglik += count * log(pdf(binomial, y));
  }
  loglik = loglik / n_sample;
  return -loglik;
}

binomial_params mle_binomial(const triplets& ind2train,
                             unsigned long int n_sample) {
  double sum = 0;
  double val;
  double max_val = -1;
  unsigned long int ri;
  unsigned long int n_train_nonzero = ind2train.size();
  itemmap val2count;
  for (size_t i = 0; i < n_sample; ++i) {
    ri = rand() % n_train_nonzero;
    ri = i;
    val = ind2train[ri].response;
    if (val2count.count(val))
      val2count[val] += 1;
    else
      val2count[val] = 1;
    sum += val;
    if (val > max_val) max_val = val;
  }
  double mean = sum / n_sample;
  pair<double, double> result = bmath::tools::brent_find_minima(
      bind(binomial_loglik, placeholders::_1, val2count, mean), max_val - 1, 100000.0, 32);
  binomial_params out;
  out.r = round(result.first);
  out.p = mean / round(result.first);
  return out;
}

double negative_binomial_loglik(double r, itemmap& val2count, double mean) {
  double loglik = 0.0;
  double p = mean / (mean + r);
  unsigned long int y, count;
  bmath::negative_binomial nb(r, p);
  itemmap::iterator it;
  unsigned long int n_sample = 0;
  for (it = val2count.begin(); it != val2count.end(); it++) {
    y = it->first;
    count = it->second;
    n_sample += count;
    loglik += count * log(pdf(nb, y));
  }
  loglik = loglik / n_sample;
  return -loglik;
}

negative_binomial_params mle_negative_binomial(const triplets& ind2train,
                                               unsigned long int n_sample) {
  double sum = 0;
  double val;
  unsigned long int ri;
  unsigned long int n_train_nonzero = ind2train.size();
  itemmap val2count;
  for (size_t i = 0; i < n_sample; ++i) {
    ri = rand() % n_train_nonzero;
    val = ind2train[ri].response;
    if (val2count.count(val))
      val2count[val] += 1;
    else
      val2count[val] = 1;
    sum += val;
  }
  double mean = sum / n_sample;
  pair<double, double> result = bmath::tools::brent_find_minima(
      bind(negative_binomial_loglik, placeholders::_1, val2count, mean), 0.0, 100000.0, 32);
  negative_binomial_params out;
  out.r = result.first;
  out.p = mean / (mean + result.first);
  return out;
}

ztp_params mle_ztp(const triplets& ind2train, unsigned long int n_sample) {
  double sum = 0;
  double val;
  unsigned long int ri;
  unsigned long int n_train_nonzero = ind2train.size();
  for (size_t i = 0; i < n_sample; ++i) {
    ri = rand() % n_train_nonzero;
    val = ind2train[ri].response;
    sum += val;
  }
  double mu = sum / n_sample;
  double lambda = lambertw(-exp(-mu) * mu) + mu;
  ztp_params out;
  out.lambda = lambda;
  return out;
}

double calc_auc(const predictions& p) {
  unsigned long int i, truePos, tp0, accum, tn, ones = 0;
  double threshold;  // predictions <= threshold are classified as zeros
  unsigned long int count = p.size();
  for (i = 0; i < count; i++) ones += p[i].label;
  if (0 == ones || count == ones) return 1;

  truePos = tp0 = ones;
  accum = tn = 0;
  threshold = p[0].pred;
  for (i = 0; i < count; i++) {
    if (p[i].pred != threshold) {  // threshold changes
      threshold = p[i].pred;
      accum += tn * (truePos + tp0);  // 2* the area of trapezoid
      tp0 = truePos;
      tn = 0;
    }
    tn += 1 - p[i].label;  // x-distance between adjacent points
    truePos -= p[i].label;
  }
  accum += tn * (truePos + tp0);  // 2* the area of trapezoid
  return (double)accum / (2 * ones * (count - ones));
}

void calc_metrics(const vector<double>& weight, predictions& p,
                  vector<double>& hit, vector<double>& recallatk,
                  vector<double>& precisionatk, vector<double>& avprecisionatk,
                  vector<double>& ndcgatk, int k_max, int n_test) {
  // p[i].pred = prediction[i]
  // p[i].label = i in y_test
  // weight[i] = 1 / log2(i+2) for i=0,1,...,k_max-1
  int i;
  partial_sort(p.begin(), p.begin() + 2*k_max, p.end());
  double norm = 1;
  int cur_ind = 0;
  while(p[cur_ind].label == -1)
    cur_ind += 1;
  hit[0] = p[cur_ind].label;
  precisionatk[0] = hit[0];
  avprecisionatk[0] = hit[0];
  recallatk[0] = hit[0];
  ndcgatk[0] = hit[0];
  for (i = 1; i < k_max; i++) {
    do{
      cur_ind += 1;
    } while(p[cur_ind].label == -1);
    hit[i] = hit[i - 1] + p[cur_ind].label;
    norm = i + 1;
    precisionatk[i] = hit[i] / norm;
    avprecisionatk[i] = avprecisionatk[i - 1] + precisionatk[i]*p[cur_ind].label;
    if (n_test < norm) norm = n_test;
    recallatk[i] = hit[i] / norm;
    if (p[cur_ind].label == 1)
      ndcgatk[i] = ndcgatk[i - 1] + weight[i];
    else
      ndcgatk[i] = ndcgatk[i - 1];
  }
  for (i = 1; i < k_max; i++) {
    norm = i + 1;
    if (n_test < norm) norm = n_test;
    avprecisionatk[i] = avprecisionatk[i] / norm;
  }
  double maxdcg = 0;
  if (k_max > n_test){
    for (i = 0; i < n_test; i++){
        maxdcg += weight[i];
        ndcgatk[i] = ndcgatk[i] / maxdcg;
    }
    for (i = n_test; i < k_max; i++)
        ndcgatk[i] = ndcgatk[i] / maxdcg;
  }
  else
    for (i = 0; i < k_max; i++){
        maxdcg += weight[i];
        ndcgatk[i] = ndcgatk[i] / maxdcg;
    }
}

void write_h5vector(H5::H5File& file, string& dsetname, vector<double>& x) {
  hsize_t dimsf[1];
  dimsf[0] = x.size();
  H5::DataSpace dspace(1, dimsf);
  H5::DataSet dset(
      file.createDataSet(dsetname, H5::PredType::NATIVE_DOUBLE, dspace));
  dset.write(x.data(), H5::PredType::NATIVE_DOUBLE);
  dset.close();
  dspace.close();
}

void write_h5vector(H5::H5File& file, string& dsetname,
                   vector<unsigned long int>& x) {
  hsize_t dimsf[1];
  dimsf[0] = x.size();
  H5::DataSpace dspace(1, dimsf);
  H5::DataSet dset(
      file.createDataSet(dsetname, H5::PredType::NATIVE_ULONG, dspace));
  dset.write(x.data(), H5::PredType::NATIVE_ULONG);
  dset.close();
  dspace.close();
}

void write_h5vector(H5::H5File& file, string& dsetname, vector<int>& x) {
  hsize_t dimsf[1];
  dimsf[0] = x.size();
  H5::DataSpace dspace(1, dimsf);
  H5::DataSet dset(
      file.createDataSet(dsetname, H5::PredType::NATIVE_INT, dspace));
  dset.write(x.data(), H5::PredType::NATIVE_INT);
  dset.close();
  dspace.close();
}

void write_h5array2d(H5::H5File& file, string& dsetname, array2d& x) {
  hsize_t dimsf[2];
  dimsf[0] = x.size();
  dimsf[1] = x[0].size();
  H5::DataSpace dspace(2, dimsf);
  H5::DataSet dset(
      file.createDataSet(dsetname, H5::PredType::NATIVE_DOUBLE, dspace));
  dset.write(x.data(), H5::PredType::NATIVE_DOUBLE);
  dset.close();
  dspace.close();
}

void read_h5vector(H5::H5File& file, string& dsetname, vector<double>& x) {
  H5::DataSet dset = file.openDataSet(dsetname);
  H5::DataSpace dspace = dset.getSpace();
  hsize_t dims[1];
  int ndims = dspace.getSimpleExtentDims(dims, NULL);
  int length = dims[0];
  dset.read(x.data(), H5::PredType::NATIVE_DOUBLE);
  dspace.close();
  dset.close();
}

void read_h5vector(H5::H5File& file, string& dsetname,
                   vector<unsigned long int>& x) {
  H5::DataSet dset = file.openDataSet(dsetname);
  H5::DataSpace dspace = dset.getSpace();
  hsize_t dims[1];
  int ndims = dspace.getSimpleExtentDims(dims, NULL);
  int length = dims[0];
  // x.resize(length);
  dset.read(x.data(), H5::PredType::NATIVE_ULONG);
  dspace.close();
  dset.close();
}

void read_h5vector(H5::H5File& file, string& dsetname, vector<int>& x) {
  H5::DataSet dset = file.openDataSet(dsetname);
  H5::DataSpace dspace = dset.getSpace();
  hsize_t dims[1];
  int ndims = dspace.getSimpleExtentDims(dims, NULL);
  int length = dims[0];
  // x.resize(length);
  dset.read(x.data(), H5::PredType::NATIVE_INT);
  dspace.close();
  dset.close();
}

void read_h5array2d(H5::H5File& file, string& dsetname, array2d& x) {
  H5::DataSet dset = file.openDataSet(dsetname);
  dset.read(x.data(), H5::PredType::NATIVE_DOUBLE);
  dset.close();
}

void add_vector_to_ptree(bptree::ptree& pt, string& key, vector<double>& value,
                         int index = -1) {
  if (index == -1) {
    index = value.size();
  }
  bptree::ptree child;
  bptree::ptree element;
  for (int i = 0; i < index; i++) {
    element.put_value(value[i]);
    child.push_back(make_pair("", element));
  }
  pt.add_child(key, child);
}
