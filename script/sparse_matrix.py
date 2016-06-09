import numpy as np
import os
import pandas as pd
import pickle

class SparseMatrix(object):

    def __init__(self, test_ratio=0.20, validation_ratio=0.01, n_eff_test_zero=0):
        '''
        Partitions data into trainining, validation and testing sets
        '''
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio
        self.users = set()
        self.items = set()
        self.np_users = None
        self.np_items = None
        self.train = dict()
        self.test = dict()
        self.val = dict()
        self.n_user = -1
        self.n_item = -1
        self.n_total_nonzero = -1
        self.n_train_nonzero = -1
        self.n_test_nonzero = -1
        self.n_val_nonzero = -1
        self.n_total_zero = -1
        self.n_train_zero = -1
        self.n_test_zero = -1
        self.n_val_zero = -1
        self.average_response = -1
        self.max_response = -1
        self.min_response = -1
        self.sparsity = -1
        self.n_eff_test_zero = 1
        self.n_eff_val_zero = 1
        self.test_zero = np.zeros([self.n_eff_test_zero, 2], dtype='int32')
        self.val_zero = np.zeros([self.n_eff_val_zero, 2], dtype='int32')

    def from_dataframe(self, df, shuffle=False):
        '''
        Constructs sparse matrix from pandas dataframe.
        Dataframe has the columns index, user (row index), item (col index), response
        '''
        self.n_total_nonzero = len(df)
        self.n_val_nonzero = int(len(df)*self.validation_ratio)
        self.n_test_nonzero = int(len(df)*self.test_ratio)
        self.n_train_nonzero = len(df) - self.n_val_nonzero - self.n_test_nonzero
        if shuffle:
            df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)
        self.train_df = df.iloc[:self.n_train_nonzero,:]
        self.val_df = df.iloc[self.n_train_nonzero:self.n_train_nonzero+self.n_val_nonzero,:]
        self.test_df = df.iloc[self.n_train_nonzero+self.n_val_nonzero:,:]
        for indi, user, item, response in df.itertuples():
            self.users.add(user)
            self.items.add(item)
            if indi < self.n_train_nonzero:
                if user not in self.train:
                    self.train[user] = {}
                self.train[user][item] = response
            elif indi < (self.n_train_nonzero + self.n_val_nonzero):
                if user not in self.val:
                    self.val[user] = {}
                self.val[user][item] = response
            else:
                if user not in self.test:
                    self.test[user] = {}
                self.test[user][item] = response
        self.n_user = len(self.users)
        self.n_item = len(self.items)
        self.np_users = np.array(list(self.users))
        self.np_items = np.array(list(self.items))
        self.n_eff_test_zero = self.n_test_nonzero
        self.test_zero = np.zeros([self.n_eff_test_zero, 2], dtype='int32')
        self.n_eff_val_zero = self.n_val_nonzero
        self.val_zero = np.zeros([self.n_eff_val_zero, 2], dtype='int32')

        max_z = 2*(self.n_eff_test_zero + self.n_eff_val_zero)
        indices = np.random.random_integers(0,
                                            self.n_user * self.n_item - 1,
                                            max_z)
        val_zero_count = 0
        test_zero_count = 0
        for indi in indices:
            item_indi = indi / self.n_user
            user_indi = indi % self.n_user
            user = self.np_users[user_indi]
            item = self.np_items[item_indi]
            if not((user in self.train and item in self.train[user])
                    or (user in self.test and item in self.test[user])
                    or (user in self.val and item in self.val[user])):
                if val_zero_count < self.n_eff_val_zero:
                    self.val_zero[val_zero_count][0] = user
                    self.val_zero[val_zero_count][1] = item
                    val_zero_count += 1
                else:
                    self.test_zero[test_zero_count][0] = user
                    self.test_zero[test_zero_count][1] = item
                    test_zero_count += 1
            if test_zero_count == self.n_eff_test_zero:
                break

    def to_pickle(self, fname):
        with open(fname, 'w') as f:
            pickle.dump(self.__dict__, f)

    def to_csv(self, base_dir, dataset):
        '''
        Creates the necessary csv files for cpf
        '''
        wd = os.path.join(base_dir, dataset)
        if not os.path.exists(wd):
            os.makedirs(wd)
        fname = os.path.join(wd, 'users.csv')
        np.random.shuffle(self.np_users)
        pd.Series(self.np_users).to_csv(fname, sep=' ', index=False)

        fname = os.path.join(wd, 'items.csv')
        np.random.shuffle(self.np_items)
        pd.Series(self.np_items).to_csv(fname, sep=' ', index=False)

        fname = os.path.join(wd, 'train.csv')
        self.train_df.to_csv(fname, header=False, sep=' ', index=False)

        fname = os.path.join(wd, 'val.csv')
        self.val_df.to_csv(fname, header=False, sep=' ', index=False)

        fname = os.path.join(wd, 'test.csv')
        self.test_df.to_csv(fname, header=False, sep=' ', index=False)

        fname = os.path.join(wd, 'test_zero.csv')
        pd.DataFrame(self.test_zero,
                     columns=['user', 'item']).to_csv(fname,
                                                      header=False, sep=' ', index=False)

        fname = os.path.join(wd, 'val_zero.csv')
        pd.DataFrame(self.val_zero,
                     columns=['user', 'item']).to_csv(fname,
                                                      header=False, sep=' ', index=False)
