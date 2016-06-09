import configparser
import os

def create_test_config():
    wd = '../test/'
    dataset = 'yelp_250k'
    cfg = configparser.RawConfigParser()
    confnum = 1
    for response_model in ['Binomial', 'Gamma', 'Inverse Gaussian', 'Negative Binomial',
                           'Poisson', 'ZTP', 'HPF', 'UserPMM', 'ItemPMM',
                           'Gaussian', 'PMF', 'UserGMM', 'ItemGMM']:
        for sparsity_model in ['None', 'HCPF', 'HICPF']:
            for sample_method in ['nonmissing', 'full', 'binary']:
                strconfnum = str(confnum)
                cfg.add_section(strconfnum)
                cfg.set(strconfnum, 'in_dir', os.path.join('./test/data/yelp_250k/'))
                cfg.set(strconfnum, 'out_dir', os.path.join('./test/data/yelp_250k/'))
                cfg.set(strconfnum, 'response_model', response_model)
                cfg.set(strconfnum, 'sparsity_model', sparsity_model)
                cfg.set(strconfnum, 'dataset', dataset)
                cfg.set(strconfnum, 'sample_method', sample_method)
                cfg.set(strconfnum, 'batchsize', 100000)
                cfg.set(strconfnum, 'test_every', 1)
                cfg.set(strconfnum, 'save_pred_every', 100)
                cfg.set(strconfnum, 'save_every', 100)
                cfg.set(strconfnum, 'max_iter', 100)
                cfg.set(strconfnum, 'n_component', 40)
                cfg.set(strconfnum, 'xi', 0.7)
                cfg.set(strconfnum, 'tau', 10000)
                cfg.set(strconfnum, 'test_ratio', 0.2)
                cfg.set(strconfnum, 'validation_ratio', 0.01)
                confnum += 1
    fname = os.path.join(wd, 'test.ini')
    with open(fname, 'w') as configfile:
        cfg.write(configfile)

if __name__ == '__main__':
    create_test_config()
