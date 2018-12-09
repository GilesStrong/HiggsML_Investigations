from __future__ import division
import numpy as np
import pandas
import math
import os
import types
import h5py
from six.moves import cPickle as pickle
from pathlib import Path

from hepml_tools.plotting_and_evaluation.plotters import *
from hepml_tools.general.misc_functions import *
from hepml_tools.general.metrics import *
from hepml_tools.general.fold_train import get_feature

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

DATA_PATH = Path("../data/")


def score_test_data(test_data, cut, pred_name='pred'):
    data = pandas.DataFrame()
    data['pred_class'] = get_feature(pred_name, test_data)
    data['gen_weight'] = get_feature('weights', test_data)
    data['gen_target'] = get_feature('targets', test_data)
    data['private'] = get_feature('private', test_data)

    accept = (data.pred_class >= cut)
    signal = (data.gen_target == 1)
    bkg = (data.gen_target == 0)
    public = (data.private == 0)
    private = (data.private == 1)

    public_ams = calc_ams(np.sum(data.loc[accept & public & signal, 'gen_weight']),
                          np.sum(data.loc[accept & public & bkg, 'gen_weight']))

    private_ams = calc_ams(np.sum(data.loc[accept & private & signal, 'gen_weight']),
                           np.sum(data.loc[accept & private & bkg, 'gen_weight']))

    print("Public:Private AMS: {} : {}".format(public_ams, private_ams))    
    return public_ams, private_ams


def export_test_to_csv(cut, name, data_path=DATA_PATH):
    test_data = h5py.File(data_path + 'testing.hdf5', "r+")

    data = pandas.DataFrame()
    data['EventId'] = get_feature('EventId', test_data)
    data['pred_class'] = get_feature('pred', test_data)

    data['Class'] = 'b'
    data.loc[data.pred_class >= cut, 'Class'] = 's'

    data.sort_values(by=['pred_class'], inplace=True)
    data['RankOrder'] = range(1, len(data) + 1)
    data.sort_values(by=['EventId'], inplace=True)

    print(data_path + name + '_test.csv')
    data.to_csv(data_path + name + '_test.csv', columns=['EventId', 'RankOrder', 'Class'], index=False)


def convert_to_df(datafile, pred_name='pred', n_load=-1, set_fold=-1):
    data = pandas.DataFrame()
    data['gen_target'] = get_feature('targets', datafile, n_load, set_fold=set_fold)
    data['gen_weight'] = get_feature('weights', datafile, n_load, set_fold=set_fold)
    data['pred_class'] = get_feature(pred_name, datafile, n_load, set_fold=set_fold)
    print(len(data), "candidates loaded")
    return data
