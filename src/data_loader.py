import os
import numpy as np
import pickle
from torch.utils import data
import ipdb

class Mtg_AudioFolder(data.Dataset):
    def __init__(self, root, subset, tr_val='train', split=0):
        self.trval = tr_val
        self.root = root
        fn = os.path.join(root, 'split', subset, 'split-'+str(split), tr_val+'dict.pickle')
        self.get_dictionary(fn)

    def __getitem__(self, index):
        fn = os.path.join(self.root, 'vqvae', self.dictionary[index]['path'][:-3]+'npy')
        audio = np.squeeze(np.array(np.load(fn)))
        tags = self.dictionary[index]['tags']
        return fn, audio.astype('float32'), tags.astype('float32')

    def get_dictionary(self, fn):
        with open(fn, 'rb') as pf:
            dictionary = pickle.load(pf)
        self.dictionary = dictionary

    def __len__(self):
        return len(self.dictionary)


def get_mtg_loader(root, subset, batch_size, tr_val='train', split=0, num_workers=1):
    data_loader = data.DataLoader(dataset=Mtg_AudioFolder(root, subset, tr_val, split),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader






class Mer31k_AudioFolder(data.Dataset):
    def __init__(self, root, tr_val='train'):
        self.trval = tr_val
        self.root = root
        fn = os.path.join(root,'dict_split', tr_val+'_dict.pickle')
        self.get_dictionary(fn)

    def __getitem__(self, index):
        fn = os.path.join(self.root, 'mel_vqvae', self.dictionary[index]['path'].split('/')[-1])
        audio = np.array(np.load(fn))
        tags = np.array(self.dictionary[index]['tags'])
        return fn, audio.astype('float32'), tags.astype('float32')

    def get_dictionary(self, fn):
        with open(fn, 'rb') as pf:
            dictionary = pickle.load(pf)
        self.dictionary = dictionary

    def __len__(self):
        return len(self.dictionary)


def get_mer31k_loader(root, batch_size, tr_val='train', num_workers=1):
    data_loader = data.DataLoader(dataset=Mer31k_AudioFolder(root, tr_val),
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader


