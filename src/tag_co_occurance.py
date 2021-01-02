import numpy as np
import os
import csv
import ipdb
import pandas as pd
from collections import Counter
import get_numpy
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

class Get_tag_co_occurrence(object): 
    def __init__(self, num_classes, save_fn, file_name, dictionary):
        self.num_classes = num_classes
        self.save_fn = save_fn
        self.file_name = file_name
        self.dictionary = dictionary
        self.num_sample = len(self.dictionary)
        self.tags = self.get_tag_list()


    def get_tag_list(self):
        tag_list = np.load('./baseline/tag_list.npy')
        tags = [tag[13:] for tag in tag_list]
        return tags


    def get_tag_occurrence(self):
        pd_ = pd.DataFrame.from_dict(self.dictionary).T
        one_tag = np.zeros(self.num_classes)
        for i in range(self.num_classes):
            for t in range(self.num_sample):
                tag = self.dictionary[t]['tags']
                if int(tag[i]) == 1 :
                    one_tag[i] += 1
        return one_tag

    def get_tuple_occurrence(self):
        tuple_o = np.zeros([self.num_classes,self.num_classes])
        for j in range(self.num_classes):
            for i in range(self.num_classes):
                for t in range(self.num_sample):
                    tag = self.dictionary[t]['tags']
                    if int(tag[j]) == 1 and int(tag[i]) == 1:
                        tuple_o[i][j] += 1
        return tuple_o

    def plot_hotspot(self, tag_co_occurrence):
        # Draw a heatmap with the numeric values in each cell
        f, ax = plt.subplots(figsize=(30, 30))
        ax.xaxis.set_ticks_position('top')
        sns.heatmap(tag_co_occurrence, annot=True, linewidths=.5, ax = ax,  
                     vmin=1, vmax=self.num_classes, cbar = False, xticklabels = self.tags, yticklabels = self.tags)
        path = os.path.join(self.save_fn, self.file_name)
        plt.savefig(path)
        print('figure save to', path)


    def get_tag_co_occurrence(self, tag_o, tuple_o):
        tag_co_occurrence = np.zeros([self.num_classes,self.num_classes])
        for j in range(self.num_classes):
            for i in range(self.num_classes):
                tag_co_occurrence[j][i] = round((tuple_o[j][i] / tag_o[i]) * 100)
        return tag_co_occurrence

    def run(self):
        tag_o   = self.get_tag_occurrence()
        tuple_o = self.get_tuple_occurrence()
        tag_co_occurrence = self.get_tag_co_occurrence(tag_o, tuple_o)
        self.plot_hotspot(tag_co_occurrence)





def check_exist(fn):
    if not os.path.exists(fn):
        os.mkdir(fn)


if __name__ == "__main__":
    save_fn = './figs'
    check_exist(save_fn)
    num_classes = 56
    dictionary_train, dictionary_val, dictionary_test = get_numpy.run(0)
    
    get_train = Get_tag_co_occurrence(num_classes, save_fn, 'train_tag_co.png', dictionary_train)
    get_train.run()

    get_val = Get_tag_co_occurrence(num_classes, save_fn, 'val_tag_co.png', dictionary_val)    
    get_val.run()

    get_test = Get_tag_co_occurrence(num_classes, save_fn, 'test_tag_co.png', dictionary_test)
    get_test.run()

