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


def str2bool(v):
    if isinstance(v, bool):
        return v
    else:
        return v.lower() in ("yes", "true", "t", "1")


def import_data(data_path=Path("../Data/"),
                rotate=False, flip_y=False, flip_z=False, cartesian=True,
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

    convert_data(training_data, rotate, flip_y, flip_z, cartesian)
    convert_data(test, rotate, flip_y, flip_z, cartesian)

    training_data['gen_target'] = 0
    training_data.loc[training_data.Label == 's', 'gen_target'] = 1
    training_data.drop(columns=['Label'], inplace=True)
    training_data['gen_weight_original'] = training_data['gen_weight']  # gen_weight might be renormalised

    # train_feats = [x for x in training_data.columns if 'gen' not in x and x != 'EventId' and 'kaggle' not in x.lower()]
    vec_feats = [x for x in training_data.columns if '_px' in x or '_py' in x or '_pz' in x]
    extra_feats = [x for x in training_data.columns if x not in vec_feats and 'gen' not in x and x != 'EventId' and 'kaggle' not in x.lower()]
    train_feats = extra_feats + vec_feats

    train, val = train_test_split(training_data, test_size=val_size, random_state=seed)

    print('Training on {} datapoints and validating on {}, using {} feats:\n{}'.format(len(train), len(val), len(train_feats), [x for x in train_feats]))

    return {'train': train[train_feats + ['gen_target', 'gen_weight', 'gen_weight_original']], 
            'val': val[train_feats + ['gen_target', 'gen_weight', 'gen_weight_original']],
            'test': test,
            'feats': {'extra': extra_feats, 'vec': vec_feats}}


def rotate_event(in_data):
    '''Rotate event in phi such that lepton is at phi == 0'''
    in_data['PRI_tau_phi'] = in_data.apply(lambda row: delta_phi(row['PRI_lep_phi'], row['PRI_tau_phi']), axis=1)
    in_data['PRI_jet_leading_phi'] = in_data.apply(lambda row: delta_phi(row['PRI_lep_phi'], row['PRI_jet_leading_phi']), axis=1)
    in_data['PRI_jet_subleading_phi'] = in_data.apply(lambda row: delta_phi(row['PRI_lep_phi'], row['PRI_jet_subleading_phi']), axis=1)
    in_data['PRI_met_phi'] = in_data.apply(lambda row: delta_phi(row['PRI_lep_phi'], row['PRI_met_phi']), axis=1)
    in_data['PRI_lep_phi'] = 0


def z_flip_event(in_data):
    '''Flip event in z-axis such that primary lepton is in positive z-direction'''
    if 'PRI_lep_eta' in in_data.columns:
        cut = (in_data.PRI_lep_eta < 0)
        
        for particle in ['PRI_lep', 'PRI_tau', 'PRI_jet_leading', 'PRI_jet_subleading']:
            in_data.loc[cut, particle + '_eta'] = -in_data.loc[cut, particle + '_eta'] 
    
    else:
        cut = (in_data.PRI_lep_pz < 0)
        
        for particle in ['PRI_lep', 'PRI_tau', 'PRI_jet_leading', 'PRI_jet_subleading']:
            in_data.loc[cut, particle + '_pz'] = -in_data.loc[cut, particle + '_pz'] 


def y_flip_event(in_data):
    '''Flip event in x-axis such that tau has a higher py than the lepton'''
    if 'PRI_tau_phi' in in_data.columns:
        cut = (in_data.PRI_tau_phi < 0)
        
        for particle in ['PRI_tau', 'PRI_jet_leading', 'PRI_jet_subleading', 'PRI_met']:
            in_data.loc[cut, particle + '_phi'] = -in_data.loc[cut, particle + '_phi'] 
    
    else:
        cut = (in_data.PRI_tau_py < 0)
        
        for particle in ['PRI_tau', 'PRI_jet_leading', 'PRI_jet_subleading', 'PRI_met']:
            in_data.loc[cut, particle + '_py'] = -in_data.loc[cut, particle + '_py'] 
    

def convert_data(in_data, rotate=False, flip_y=False, flip_z=False, cartesian=False):
    '''Pass data through conversions and drop uneeded columns'''
    in_data.replace([np.inf, -np.inf, -999.0], np.nan, inplace=True)
    
    if rotate:
        print('Setting lepton to phi = 0')
        rotate_event(in_data)
        
        if flip_y:
            print('Setting tau to positve y')
            y_flip_event(in_data)

    if flip_z:
        print('Setting lepton positive z')
        z_flip_event(in_data)
            
    if cartesian:
        print("Converting to Cartesian coordinates")
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
    
    X = np.hstack((input_pipe['extra'].transform(in_data[feats['extra']].values.astype('float32')),
                  input_pipe['vec'].transform(in_data[feats['vec']].values.astype('float32'))))
     
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


def run_data_import(data_path, rotate, flip_y, flip_z, cartesian, mode, val_size, seed, n_folds, vec_mean):
    '''Run through all the stages to save the data into files for training, validation, and testing'''
    # Get Data
    data = import_data(data_path, rotate, flip_y, flip_z, cartesian, mode, val_size, seed)

    # Standardise and normalise
    input_pipe = {}
    input_pipe['extra'], _ = get_pre_proc_pipes(norm_in=True)
    input_pipe['extra'].fit(data['train'][data['feats']['extra']].values.astype('float32'))

    input_pipe['vec'], _ = get_pre_proc_pipes(norm_in=True, with_mean=vec_mean)
    input_pipe['vec'].fit(data['train'][data['feats']['vec']].values.astype('float32'))
    
    for pipe in input_pipe:
        with open(data_path + f'input_pipe_{pipe}.pkl', 'wb') as fout:
            pickle.dump(input_pipe[pipe], fout)

    prepare_sample(data['train'], 'train', input_pipe, True, n_folds, data['feats'], data_path)
    prepare_sample(data['val'], 'val', input_pipe, False, n_folds, data['feats'], data_path)
    prepare_sample(data['test'], 'testing', input_pipe, False, n_folds, data['feats'], data_path)

    with open(data_path + 'feats.pkl', 'wb') as fout:
        pickle.dump(data['feats'], fout)


if __name__ == '__main__':
    parser = optparse.OptionParser(usage=__doc__)
    parser.add_option("-d", "--data_path", dest="data_path", action="store", default="./data/", help="Data folder location")
    parser.add_option("-r", "--rotate", dest="rotate", action="store", default=False, help="Rotate events in phi to have common alignment")
    parser.add_option("-y", "--flipy", dest="flip_y", action="store", default=False, help="Flip events in y to have common alignment")
    parser.add_option("-z", "--flipz", dest="flip_z", action="store", default=False, help="Flip events in z to have common alignment")
    parser.add_option("-c", "--cartesian", dest="cartesian", action="store", default=True, help="Convert to Cartesian system")
    parser.add_option("-m", "--mode", dest="mode", action="store", default="OpenData", help="Using open data or Kaggle data")
    parser.add_option("-v", "--val_size", dest="val_size", action="store", default=0.2, help="Fraction of data to use for validation")
    parser.add_option("-s", "--seed", dest="seed", action="store", default=1337, help="Seed for train/val split")
    parser.add_option("-n", "--n_folds", dest="n_folds", action="store", default=10, help="Number of folds to split data")
    parser.add_option("-zv", "--zero_vec", dest="vec_mean", action="store", default=True, help="Set vector means to zero")
    opts, args = parser.parse_args()

    run_data_import(opts.data_path,
                    str2bool(opts.rotate), str2bool(opts.flip_y), str2bool(opts.flip_z), str2bool(opts.cartesian),
                    opts.mode, opts.val_size, opts.seed, opts.n_folds, str2bool(opts.zero_vec))
    