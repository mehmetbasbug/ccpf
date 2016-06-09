#include "utils.h"
#include "base.h"
#include "sparse_matrix.h"
#include "hcpf.h"
#include "ccpf.h"
#include "element/binomial.h"
#include "element/gamma.h"
#include "element/gaussian.h"
#include "element/inverse_gaussian.h"
#include "element/negative_binomial.h"
#include "element/poisson.h"
#include "element/ztp.h"
#include "mixture/gmm.h"
#include "mixture/pmm.h"
#include "factor/hpf.h"
#include "factor/pmf.h"
#include "regression/regression.h"
#include <boost/program_options.hpp>
using namespace std;
namespace bpo = boost::program_options;

int main(int argc, char* argv[]) {
  try {
    bpo::options_description desc("Configuration");
    desc.add_options()("help", "produce help message")(
        "fname", bpo::value<string>()->required(), "Conf file")(
        "conf_num", bpo::value<unsigned long int>()->required(), "Conf num")(
        "post", bpo::value<bool>()->default_value(0),
        "Whether to perform post analysis")(
        "save", bpo::value<bool>()->default_value(0),
        "Whether to save test predictions");

    bpo::variables_map vm;
    bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
    bpo::notify(vm);

    if (vm.count("help")) {
      cout << desc << "\n";
      return 0;
    }
    string fname = vm["fname"].as<string>();
    unsigned long int conf_num = vm["conf_num"].as<unsigned long int>();
    Configuration cfg;
    cfg.read(fname, conf_num);
    cout << "Config file is read.\n";
    bool post = vm["post"].as<bool>();
    bool save = vm["save"].as<bool>();

    srand((unsigned)time(0));
    SparseMatrix sm((cfg.get_in_dir() + "users.csv").c_str(),
                    (cfg.get_in_dir() + "items.csv").c_str(),
                    (cfg.get_in_dir() + "train.csv").c_str(),
                    (cfg.get_in_dir() + "val.csv").c_str(),
                    (cfg.get_in_dir() + "test.csv").c_str(),
                    (cfg.get_in_dir() + "val_zero.csv").c_str(),
                    (cfg.get_in_dir() + "test_zero.csv").c_str(),
                    cfg.get_validation_ratio(), cfg.get_test_ratio());
    string smdlstr = cfg.get_sparsity_model();
    string rmdlstr = cfg.get_response_model();
    int n_component_ = cfg.get_n_component();
    double xi_ = cfg.get_xi();
    double tau_ = cfg.get_tau();
    int n_user_max = 1000;
    int n_trunc_ = 10;  // This is overriden
    if (smdlstr.compare("None") == 0) {
      Model* mdl;
      if (rmdlstr.compare("Binomial") == 0)
        mdl = new Binomial();
      else if (rmdlstr.compare("Gamma") == 0)
        mdl = new Gamma();
      else if (rmdlstr.compare("Inverse Gaussian") == 0)
        mdl = new InverseGaussian();
      else if (rmdlstr.compare("Negative Binomial") == 0)
        mdl = new NegativeBinomial();
      else if (rmdlstr.compare("Poisson") == 0)
        mdl = new Poisson();
      else if (rmdlstr.compare("ZTP") == 0)
        mdl = new ZTP();
      else if (rmdlstr.compare("HPF") == 0)
        mdl = new HPF();
      else if (rmdlstr.compare("UserPMM") == 0)
        mdl = new UserPMM();
      else if (rmdlstr.compare("ItemPMM") == 0)
        mdl = new ItemPMM();
      else if (rmdlstr.compare("Gaussian") == 0)
        mdl = new Gaussian();
      else if (rmdlstr.compare("PMF") == 0)
        mdl = new PMF();
      else if (rmdlstr.compare("UserGMM") == 0)
        mdl = new UserGMM();
      else if (rmdlstr.compare("ItemGMM") == 0)
        mdl = new ItemGMM();
      else if (rmdlstr.compare("Regression") == 0)
        mdl = new Regression();
      else
        cerr << "Unknown Model." << endl;
      mdl->initialize_with_sparse_matrix(sm, n_component_, n_trunc_, xi_, tau_);
      Result result(cfg, sm, mdl->get_n_trunc());
      mdl->fit(sm, cfg.get_batchsize(), cfg.get_max_iter(),
               cfg.get_test_every(), cfg.get_save_every(),
               cfg.get_save_pred_every(), cfg.get_sample_method(),
               cfg.get_out_dir(), result, n_user_max);
    } else {
      if ((rmdlstr.compare("Binomial") == 0) ||
          (rmdlstr.compare("Gamma") == 0) ||
          (rmdlstr.compare("Inverse Gaussian") == 0) ||
          (rmdlstr.compare("Negative Binomial") == 0) ||
          (rmdlstr.compare("Poisson") == 0) ||
          (rmdlstr.compare("ZTP") == 0) ||
          (rmdlstr.compare("HPF") == 0) ||
          (rmdlstr.compare("UserPMM") == 0) ||
          (rmdlstr.compare("ItemPMM") == 0)) {
        HPFCouplingInterface* rmdl;
        if (rmdlstr.compare("Binomial") == 0)
          rmdl = new Binomial();
        else if (rmdlstr.compare("Gamma") == 0)
          rmdl = new Gamma();
        else if (rmdlstr.compare("Inverse Gaussian") == 0)
          rmdl = new InverseGaussian();
        else if (rmdlstr.compare("Negative Binomial") == 0)
          rmdl = new NegativeBinomial();
        else if (rmdlstr.compare("Poisson") == 0)
          rmdl = new Poisson();
        else if (rmdlstr.compare("ZTP") == 0)
          rmdl = new ZTP();
        else if (rmdlstr.compare("HPF") == 0)
          rmdl = new HPF();
        else if (rmdlstr.compare("UserPMM") == 0)
          rmdl = new UserPMM();
        else if (rmdlstr.compare("ItemPMM") == 0)
          rmdl = new ItemPMM();
        if (smdlstr.compare("HCPF") == 0) {
          HCPF* smdl = new HCPF();
          CCPF* mdl = new CCPF();
          mdl->set_rmdl(rmdl);
          mdl->set_smdl(smdl);
          mdl->initialize_with_sparse_matrix(sm, n_component_, n_trunc_, xi_,
                                             tau_);
          Result result(cfg, sm, mdl->get_n_trunc());
          mdl->fit(sm, cfg.get_batchsize(), cfg.get_max_iter(),
                   cfg.get_test_every(), cfg.get_save_every(),
                   cfg.get_save_pred_every(), cfg.get_sample_method(),
                   cfg.get_out_dir(), result, n_user_max);
        } else if (smdlstr.compare("HICPF") == 0) {
          HICPF* smdl = new HICPF();
          CCPF* mdl = new CCPF();
          mdl->set_rmdl(rmdl);
          mdl->set_smdl(smdl);
          mdl->initialize_with_sparse_matrix(sm, n_component_, n_trunc_, xi_,
                                             tau_);
          Result result(cfg, sm, mdl->get_n_trunc());
          mdl->fit(sm, cfg.get_batchsize(), cfg.get_max_iter(),
                   cfg.get_test_every(), cfg.get_save_every(),
                   cfg.get_save_pred_every(), cfg.get_sample_method(),
                   cfg.get_out_dir(), result, n_user_max);
        } else
          cerr << "Unknown Model." << endl;
      } else if ((rmdlstr.compare("Gaussian") == 0) ||
                 (rmdlstr.compare("PMF") == 0) ||
                 (rmdlstr.compare("UserGMM") == 0) ||
                 (rmdlstr.compare("ItemGMM") == 0) ||
                 (rmdlstr.compare("Regression") == 0)) {
        HPFNormalCouplingInterface* rmdl;
        if (rmdlstr.compare("Gaussian") == 0)
          rmdl = new Gaussian();
        else if (rmdlstr.compare("PMF") == 0)
          rmdl = new PMF();
        else if (rmdlstr.compare("UserGMM") == 0)
          rmdl = new UserGMM();
        else if (rmdlstr.compare("ItemGMM") == 0)
          rmdl = new ItemGMM();
        else if (rmdlstr.compare("Regression") == 0)
          rmdl = new Regression();
        else
          cerr << "Unknown Model." << endl;
        if (smdlstr.compare("HCPF") == 0) {
          HCPFNormal* smdl = new HCPFNormal();
          CCPFNormal* mdl = new CCPFNormal();
          mdl->set_rmdl(rmdl);
          mdl->set_smdl(smdl);
          mdl->initialize_with_sparse_matrix(sm, n_component_, n_trunc_, xi_,
                                             tau_);
          if (rmdlstr.compare("Regression") == 0){
            string xregfname = cfg.get_in_dir() + "xreg.h5";
            dynamic_cast<Regression*>(rmdl)->load_xreg(xregfname);
            dynamic_cast<Regression*>(rmdl)->reset_coef();
          }
          Result result(cfg, sm, mdl->get_n_trunc());
          mdl->fit(sm, cfg.get_batchsize(), cfg.get_max_iter(),
                   cfg.get_test_every(), cfg.get_save_every(),
                   cfg.get_save_pred_every(), cfg.get_sample_method(),
                   cfg.get_out_dir(), result, n_user_max);
        } else if (smdlstr.compare("HICPF") == 0) {
          HICPFNormal* smdl = new HICPFNormal();
          CCPFNormal* mdl = new CCPFNormal();
          mdl->set_rmdl(rmdl);
          mdl->set_smdl(smdl);
          mdl->initialize_with_sparse_matrix(sm, n_component_, n_trunc_, xi_,
                                             tau_);
          if (rmdlstr.compare("Regression") == 0){
            string xregfname = cfg.get_in_dir() + "xreg.h5";
            dynamic_cast<Regression*>(rmdl)->load_xreg(xregfname);
            dynamic_cast<Regression*>(rmdl)->reset_coef();
          }
          Result result(cfg, sm, mdl->get_n_trunc());
          mdl->fit(sm, cfg.get_batchsize(), cfg.get_max_iter(),
                   cfg.get_test_every(), cfg.get_save_every(),
                   cfg.get_save_pred_every(), cfg.get_sample_method(),
                   cfg.get_out_dir(), result, n_user_max);
        } else
          cerr << "Unknown Model." << endl;
      } else
        cerr << "Unknown Model." << endl;
    }
  } catch (exception& e) {
    cerr << "error: " << e.what() << "\n";
    return 1;
  } catch (...) {
    cerr << "Exception of unknown type!\n";
  }
  return 0;
}
