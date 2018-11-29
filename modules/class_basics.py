from hepml_tools.general.fold_train import *
from hepml_tools.general.models import getModel
from hepml_tools.general.fold_yielder import *
from hepml_tools.general.activations import *


class RotationReflectionFold(FoldYielder):
    def __init__(self, header, datafile=None, input_pipe=None,
                 rotate=True, reflect=True, rot_mult=4,
                 train_time_aug=True, test_time_aug=True):
        self.header = header
        self.rotate_aug = rotate
        self.reflect_aug = reflect
        self.augmented = True
        self.rot_mult = rot_mult
        
        if self.rotate_aug and not self.reflect_aug:
            self.aug_mult = self.rot_mult
            
        elif not self.rotate_aug and self.reflect_aug:
            self.reflectAxes = ['_px', '_py', '_pz']
            self.aug_mult = 8
            
        elif not self.rotate_aug and not self.reflect_aug:
            self.augmented = False
            train_time_aug = False
            test_time_aug = False
            self.aug_mult = 0
            print('No augmentation specified!')
            input_pipe = None
            self.getTestFold = self.getFold
            
        else:  # Reflect and rotate
            self.reflectAxes = ['_px', '_pz']
            self.aug_mult = self.rot_mult * 4
            
        self.train_time_aug = train_time_aug
        self.test_time_aug = test_time_aug
        self.input_pipe = input_pipe
        
        if not isinstance(datafile, type(None)):
            self.addSource(datafile)
    
    def rotate(self, in_data, vectors):
        for vector in vectors:
            if 'jet_leading' in vector:
                cut = in_data.PRI_jet_num >= 0.9
                in_data.loc[cut, vector + '_px'] = in_data.loc[cut, vector + '_px'] * np.cos(in_data.loc[cut, 'aug_angle']) - in_data.loc[:, vector + '_py'] * np.sin(in_data.loc[cut, 'aug_angle'])
                in_data.loc[cut, vector + '_py'] = in_data.loc[cut, vector + '_py'] * np.cos(in_data.loc[cut, 'aug_angle']) + in_data.loc[:, vector + '_px'] * np.sin(in_data.loc[cut, 'aug_angle'])

            elif 'jet_subleading' in vector:
                cut = in_data.PRI_jet_num >= 1.9
                in_data.loc[cut, vector + '_px'] = in_data.loc[cut, vector + '_px'] * np.cos(in_data.loc[cut, 'aug_angle']) - in_data.loc[:, vector + '_py'] * np.sin(in_data.loc[cut, 'aug_angle'])
                in_data.loc[cut, vector + '_py'] = in_data.loc[cut, vector + '_py'] * np.cos(in_data.loc[cut, 'aug_angle']) + in_data.loc[:, vector + '_px'] * np.sin(in_data.loc[cut, 'aug_angle'])
            
            else:
                in_data.loc[:, vector + '_px'] = in_data.loc[:, vector + '_px'] * np.cos(in_data.loc[:, 'aug_angle']) - in_data.loc[:, vector + '_py'] * np.sin(in_data.loc[:, 'aug_angle'])
                in_data.loc[:, vector + '_py'] = in_data.loc[:, vector + '_py'] * np.cos(in_data.loc[:, 'aug_angle']) + in_data.loc[:, vector + '_px'] * np.sin(in_data.loc[:, 'aug_angle'])
    
    def reflect(self, in_data, vectors):
        for vector in vectors:
            for coord in self.reflectAxes:
                try:
                    cut = (in_data['aug' + coord] == 1)
                    if 'jet_leading' in vector:
                        cut = cut & (in_data.PRI_jet_num >= 0.9)
                    elif 'jet_subleading' in vector:
                        cut = cut & (in_data.PRI_jet_num >= 1.9)
                    in_data.loc[cut, vector + coord] = -in_data.loc[cut, vector + coord]
                except KeyError:
                    pass
            
    def getFold(self, index, datafile=None):
        if isinstance(datafile, type(None)):
            datafile = self.source
            
        index = str(index)
        weights = None
        targets = None
        if 'fold_' + index + '/weights' in datafile:
            weights = np.array(datafile['fold_' + index + '/weights'])
        if 'fold_' + index + '/targets' in datafile:
            targets = np.array(datafile['fold_' + index + '/targets'])
            
        if not self.augmented:
            return {'inputs': np.array(datafile['fold_' + index + '/inputs']),
                    'targets': targets,
                    'weights': weights}

        if isinstance(self.input_pipe, type(None)):
            inputs = pandas.DataFrame(np.array(datafile['fold_' + index + '/inputs']), columns=self.header)
        else:
            inputs = pandas.DataFrame(self.input_pipe.inverse_transform(np.array(datafile['fold_' + index + '/inputs'])), columns=self.header)            
        
        vectors = [x[:-3] for x in inputs.columns if '_px' in x]
        if self.rotate_aug:
            inputs['aug_angle'] = 2 * np.pi * np.random.random(size=len(inputs))
            self.rotate(inputs, vectors)
            
        if self.reflect_aug:
            for coord in self.reflectAxes:
                inputs['aug' + coord] = np.random.randint(0, 2, size=len(inputs))
            self.reflect(inputs, vectors)
            
        if isinstance(self.input_pipe, type(None)):
            inputs = inputs[self.header].values
        else:
            inputs = self.input_pipe.transform(inputs[self.header].values)
        
        return {'inputs': inputs,
                'targets': targets,
                'weights': weights}
    
    def getTestFold(self, index, aug_index, datafile=None):
        if aug_index >= self.aug_mult:
            print("Invalid augmentation index passed", aug_index)
            return -1
        
        if isinstance(datafile, type(None)):
            datafile = self.source
            
        index = str(index)
        weights = None
        targets = None
        if 'fold_' + index + '/weights' in datafile:
            weights = np.array(datafile['fold_' + index + '/weights'])
        if 'fold_' + index + '/targets' in datafile:
            targets = np.array(datafile['fold_' + index + '/targets'])
            
        if isinstance(self.input_pipe, type(None)):
            inputs = pandas.DataFrame(np.array(datafile['fold_' + index + '/inputs']), columns=self.header)
        else:
            inputs = pandas.DataFrame(self.input_pipe.inverse_transform(np.array(datafile['fold_' + index + '/inputs'])), columns=self.header)            
            
        if self.reflect_aug and self.rotate_aug:
            rotIndex = aug_index % self.rot_mult
            refIndex = '{0:02b}'.format(int(aug_index / 4))
            vectors = [x[:-3] for x in inputs.columns if '_px' in x]
            inputs['aug_angle'] = np.linspace(0, 2 * np.pi, (self.rot_mult) + 1)[rotIndex]
            for i, coord in enumerate(self.reflectAxes):
                inputs['aug' + coord] = int(refIndex[i])
            self.rotate(inputs, vectors)
            self.reflect(inputs, vectors)
            
        elif self.reflect_aug:
            refIndex = '{0:03b}'.format(int(aug_index))
            vectors = [x[:-3] for x in inputs.columns if '_px' in x]
            for i, coord in enumerate(self.reflectAxes):
                inputs['aug' + coord] = int(refIndex[i])
            self.reflect(inputs, vectors)
            
        else:
            vectors = [x[:-3] for x in inputs.columns if '_px' in x]
            inputs['aug_angle'] = np.linspace(0, 2 * np.pi, (self.rot_mult) + 1)[aug_index]
            self.rotate(inputs, vectors)
            
        if isinstance(self.input_pipe, type(None)):
            inputs = inputs[self.header].values
        else:
            inputs = self.input_pipe.transform(inputs[self.header].values)

        return {'inputs': inputs,
                'targets': targets,
                'weights': weights}
