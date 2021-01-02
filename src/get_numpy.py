import os
import csv
import pickle
import numpy as np
from collections import Counter






def read_tsv(fn):
    r = []
    with open(fn) as tsv:
        reader = csv.reader(tsv, delimiter='\t')
        for row in reader:
            r.append(row)
    return r[1:]

def get_tag_list(path):
    rows = read_tsv(os.path.join(path, 'autotagging_moodtheme-train.tsv'))    
    option = path.split('/')[-2]
    t = []
    for row in rows:
        tags = row[5:]
        for tag in tags:
            t.append(tag)
    if option == 'top50':
        t_counter = Counter(t)
        t_sort = t_counter.most_common()[:50]
        t = [line[0] for line in t_sort]
    
    t = list(set(t))
    t.sort()
    return t
    

def get_npy_array(path, tag_list, type_='train'):
    tsv_fn = os.path.join(path, 'autotagging_moodtheme-'+type_+'.tsv')
    option = 'moodtheme'
    rows = read_tsv(tsv_fn)
    dictionary = {}
    i = 0
    for row in rows:
        temp_dict = {}
        temp_dict['artist'] = int(row[1][7:])
        temp_dict['album'] = int(row[2][6:])
        temp_dict['path'] = row[3]
        temp_dict['duration'] = (float(row[4]) * 16000 - 512) // 256
        temp_dict['tags'] = np.zeros(56)
        tags = row[5:]
        for tag in tags:
            try:
                temp_dict['tags'][tag_list.index(tag)] = 1
            except:
                continue
        if temp_dict['tags'].sum() > 0:
            dictionary[i] = temp_dict
            i += 1
    return dictionary


def run(split, option='mood'):
    PATH = '/mnt/md0/user_annahung/auto_tagging/mtg-jamendo-dataset/scripts/baseline/'
    path = os.path.join(PATH,'split',option, 'split-' + str(split))
    tag_list = get_tag_list(path)
    np.save(open(os.path.join(path, 'tag_list.npy'), 'wb'), tag_list)
    dictionary_train = get_npy_array(path, tag_list, type_='train')
    dictionary_val = get_npy_array(path, tag_list, type_='validation')
    dictionary_test = get_npy_array(path, tag_list, type_='test')
    return dictionary_train, dictionary_val, dictionary_test



    if __name__ == "__main__":
        dictionary_train, dictionary_val, dictionary_test = run(0)