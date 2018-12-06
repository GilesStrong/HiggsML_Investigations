import pandas
from six.moves import cPickle as pickle
import numpy as np
import optparse
import os
import h5py
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

from hepml_tools.transformations.hep_proc import *
from hepml_tools.general.pre_proc import get_pre_proc_pipes


def import_data(data_path=Path("../Data/"),
                rotate=False, flip_x=False, flip_z=False, cartesian=True,
                mode='OpenData',
                val_size=0.2, seed=None):
    '''Import and split data from CSV(s)'''
    if mode == 'OpenData':  # If using data from CERN Open Access
        data = pandas.read_csv(data_path + 'atlas-higgs-challenge-2014-v2.csv')
        data.rename(index=str, columns={"KaggleWeight": "gen_weight", 'PRI_met': 'PRI_met_pt'}, inplace=True)
        data.drop(columns=['Weight'], inplace=True)
        training_data = pandas.DataFrame(data.loc[data.KaggleSet == 't'])
        training_data.drop(columns=['KaggleSet'], inplace=True)
        
        test = pandas.DataFrame(data.loc[(data.KaggleSet == 'b') | (data.KaggleSet == 'v')])
        test['private'] = 0
        test.loc[(data.KaggleSet == 'v'), 'private'] = 1
        test['gen_target'] = 0
        test.loc[test.Label == 's', 'gen_target'] = 1
        test.drop(columns=['KaggleSet', 'Label'], inplace=True)

    else:  # If using data from Kaggle
        training_data = pandas.read_csv(data_path + 'training.csv')
        training_data.rename(index=str, columns={"Weight": "gen_weight", 'PRI_met': 'PRI_met_pt'}, inplace=True)
        test = pandas.read_csv(data_path + 'test.csv')
        test.rename(index=str, columns={'PRI_met': 'PRI_met_pt'}, inplace=True)

    convert_data(training_data, rotate, flip_x, flip_z, cartesian)
    convert_data(test, rotate, flip_x, flip_z, cartesian)

    training_data['gen_target'] = 0
    training_data.loc[training_data.Label == 's', 'gen_target'] = 1
    training_data.drop(columns=['Label'], inplace=True)
    training_data['gen_weight_original'] = training_data['gen_weight']  # gen_weight might be renormalised

    train_feats = [x for x in training_data.columns if 'gen' not in x and x != 'EventId' and 'kaggle' not in x.lower()]
    train, val = train_test_split(training_data, test_size=val_size, random_state=seed)

    print('Training on {} datapoints and validating on {}, using {} feats:\n{}'.format(len(train), len(val), len(train_feats), [x for x in train_feats]))

    return {'train': train[train_feats + ['gen_target', 'gen_weight', 'gen_weight_original']], 
            'val': val[train_feats + ['gen_target', 'gen_weight', 'gen_weight_original']],
            'test': test,
            'feats': train_feats}


def rotate_event(in_data):
    '''Rotate event in phi such that lepton is at phi == 0'''
    # in_data['PRI_tau_phi'] = delta_phi(in_data['PRI_lep_phi'], in_data['PRI_tau_phi'])
    # in_data['PRI_jet_leading_phi'] = delta_phi(in_data['PRI_lep_phi'], in_data['PRI_jet_leading_phi'])
    # in_data['PRI_jet_subleading_phi'] = delta_phi(in_data['PRI_lep_phi'], in_data['PRI_jet_subleading_phi'])
    # in_data['PRI_met_phi'] = delta_phi(in_data['PRI_lep_phi'], in_data['PRI_met_phi'])
    # in_data['PRI_lep_phi'] = 0
    in_data['PRI_tau_phi'] = in_data.apply(lambda row: delta_phi(row['PRI_lep_phi'], row['PRI_tau_phi']), axis=1)
    in_data['PRI_jet_leading_phi'] = in_data.apply(lambda row: delta_phi(row['PRI_lep_phi'], row['PRI_jet_leading_phi']), axis=1)
    in_data['PRI_jet_subleading_phi'] = in_data.apply(lambda row: delta_phi(row['PRI_lep_phi'], row['PRI_jet_subleading_phi']), axis=1)
    in_data['PRI_met_phi'] = in_data.apply(lambda row: delta_phi(row['PRI_lep_phi'], row['PRI_met_phi']), axis=1)
    in_data['PRI_lep_phi'] = 0


def z_flip_event(in_data):
    '''Flip event in z-axis such that primary lepton is in positive z-direction'''
    cut = (in_data.PRI_lep_eta < 0)
    
    for particle in ['PRI_lep', 'PRI_tau', 'PRI_jet_leading', 'PRI_jet_subleading']:
        in_data.loc[cut, particle + '_eta'] = -in_data.loc[cut, particle + '_eta'] 


def x_flip_event(in_data):
    '''Flip event in x-axis such that tau is in positive x-direction'''
    cut = (in_data.PRI_tau_phi < 0)
    
    for particle in ['PRI_tau', 'PRI_jet_leading', 'PRI_jet_subleading', 'PRI_met']:
        in_data.loc[cut, particle + '_phi'] = -in_data.loc[cut, particle + '_phi'] 
    

def convert_data(in_data, rotate=False, flip_x=False, flip_z=False, cartesian=True):
    '''Pass data through conversions and drop uneeded columns'''
    in_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    in_data.fillna(-999.0, inplace=True)
    in_data.replace(-999.0, 0.0, inplace=True)
    
    if rotate:
        rotate_event(in_data)
    if flip_z:
        z_flip_event(in_data)
    if flip_x:
        x_flip_event(in_data)
    
    if cartesian:
        move_to_cartesian(in_data, 'PRI_tau', drop=True)
        move_to_cartesian(in_data, 'PRI_lep', drop=True)
        move_to_cartesian(in_data, 'PRI_jet_leading', drop=True)
        move_to_cartesian(in_data, 'PRI_jet_subleading', drop=True)
        move_to_cartesian(in_data, 'PRI_met', z=False)
        
        in_data.drop(columns=["PRI_met_phi"], inplace=True)
        
    if rotate and not cartesian:
        in_data.drop(columns=["PRI_lep_phi"], inplace=True)
    elif rotate and cartesian:
        in_data.drop(columns=["PRI_lep_py"], inplace=True)


def save_fold(in_data, n, input_pipe, out_file, norm_weights, mode, feats):
    '''Save fold into hdf5 file'''
    grp = out_file.create_group('fold_' + str(n))
    
    X = input_pipe.transform(in_data[feats].values.astype('float32'))
    inputs = grp.create_dataset("inputs", shape=X.shape, dtype='float32')
    inputs[...] = X
    
    if mode != 'testing':
        if norm_weights:
            in_data.loc[in_data.gen_target == 0, 'gen_weight'] = in_data.loc[in_data.gen_target == 0, 'gen_weight'] / np.sum(in_data.loc[in_data.gen_target == 0, 'gen_weight'])
            in_data.loc[in_data.gen_target == 1, 'gen_weight'] = in_data.loc[in_data.gen_target == 1, 'gen_weight'] / np.sum(in_data.loc[in_data.gen_target == 1, 'gen_weight'])

        y = in_data['gen_target'].values.astype('int')
        targets = grp.create_dataset("targets", shape=y.shape, dtype='int')
        targets[...] = y

        X_weights = in_data['gen_weight'].values.astype('float32')
        weights = grp.create_dataset("weights", shape=X_weights.shape, dtype='float32')
        weights[...] = X_weights

        X_orig_weights = in_data['gen_weight_original'].values.astype('float32')
        orig_weights = grp.create_dataset("orig_weights", shape=X_weights.shape, dtype='float32')
        orig_weights[...] = X_orig_weights
    
    else:
        X_EventId = in_data['EventId'].values.astype('int')
        EventId = grp.create_dataset("EventId", shape=X_EventId.shape, dtype='int')
        EventId[...] = X_EventId

        if 'private' in in_data.columns:
            X_weights = in_data['gen_weight'].values.astype('float32')
            weights = grp.create_dataset("weights", shape=X_weights.shape, dtype='float32')
            weights[...] = X_weights

            X_set = in_data['private'].values.astype('int')
            KaggleSet = grp.create_dataset("private", shape=X_set.shape, dtype='int')
            KaggleSet[...] = X_set

            y = in_data['gen_target'].values.astype('int')
            targets = grp.create_dataset("targets", shape=y.shape, dtype='int')
            targets[...] = y


def prepare_sample(in_data, mode, input_pipe, norm_weights, N, feats, data_path):
    '''Split data sample into folds and save to hdf5'''
    print("Running", mode)
    os.system('rm ' + data_path + mode + '.hdf5')
    out_file = h5py.File(data_path + mode + '.hdf5', "w")

    if mode != 'testing':
        kf = StratifiedKFold(n_splits=N, shuffle=True)
        folds = kf.split(in_data, in_data['gen_target'])
    else:
        kf = KFold(n_splits=N, shuffle=True)
        folds = kf.split(in_data)

    for i, (_, fold) in enumerate(folds):
        print("Saving fold:", i, "of", len(fold), "events")
        save_fold(in_data.iloc[fold].copy(), i, input_pipe, out_file, norm_weights, mode, feats)


def run_data_import(data_path, rotate, flip_x, flip_z, cartesian, mode, val_size, seed, n_folds):
    '''Run through all the stages to save the data into files for training, validation, and testing'''
    # Get Data
    data = import_data(data_path, rotate, flip_x, flip_z, cartesian, mode, val_size, seed)

    # Standardise and normalise
    input_pipe, _ = get_pre_proc_pipes(norm_in=True)
    input_pipe.fit(data['train'][data['feats']].values.astype('float32'))
    with open(data_path + 'input_pipe.pkl', 'wb') as fout:
        pickle.dump(input_pipe, fout)

    prepare_sample(data['train'], 'train', input_pipe, True, n_folds, data['feats'], data_path)
    prepare_sample(data['val'], 'val', input_pipe, False, n_folds, data['feats'], data_path)
    prepare_sample(data['test'], 'testing', input_pipe, False, n_folds, data['feats'], data_path)

    with open(data_path + 'feats.pkl', 'wb') as fout:
        pickle.dump(data['feats'], fout)


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=__doc__)
    parser.add_option("-d", "--data_path", dest="data_path", action="store", default="./data/", help="Data folder location")
    parser.add_option("-r", "--rotate", dest="rotate", action="store", default=False, help="Rotate events in phi to have common alignment")
    parser.add_option("-x", "--flipx", dest="flip_x", action="store", default=False, help="Flip events in x to have common alignment")
    parser.add_option("-z", "--flipz", dest="flip_z", action="store", default=False, help="Flip events in z to have common alignment")
    parser.add_option("-c", "--cartesian", dest="cartesian", action="store", default=True, help="Convert to Cartesian system")
    parser.add_option("-m", "--mode", dest="mode", action="store", default="OpenData", help="Using open data or Kaggle data")
    parser.add_option("-v", "--val_size", dest="val_size", action="store", default=0.2, help="Fraction of data to use for validation")
    parser.add_option("-s", "--seed", dest="seed", action="store", default=1337, help="Seed for train/val split")
    parser.add_option("-n", "--n_folds", dest="n_folds", action="store", default=10, help="Number of folds to split data")
    opts, args = parser.parse_args()

    run_data_import(opts.data_path, opts.rotate, opts.flip_x, opts.flip_z, opts.cartesian, opts.mode, opts.val_size, opts.seed, opts.n_folds)
    