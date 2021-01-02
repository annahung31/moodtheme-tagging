import numpy as np
import glob
import os
import ipdb
from argparse import ArgumentParser
import madmom
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
import matplotlib.pyplot as plt




def compute_beat_curve(audio_path):
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[3, 4], fps=200)
    act = RNNDownBeatProcessor()
    beat_curve = proc(act(audio_path))[:,0]
    
    return beat_curve

def check_exist(folder_path):
    if not os.path.exists(folder_path):
        
        os.mkdir(folder_path)


def plot_curve(beat_curve):
    x = np.linspace(1,beat_curve.shape[0],beat_curve.shape[0])
    plt.figure()
    plt.plot(x, beat_curve)
    plt.save('beat_curve.png')     

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-f', '--folder', default='00', type=str,
                        help='folder name')
    args = parser.parse_args()
    PATH = '/mnt/md0/user_annahung/auto_tagging/mtg-jamendo-dataset/scripts/baseline'
    audio_path = glob.glob(os.path.join(PATH, 'audio' , args.folder,'*.mp3'))
    print(len(audio_path))
    for audio_ in audio_path:
        folder_path = os.path.join(PATH, 'beat', args.folder)
        file_name = audio_.split('/')[-1][:-4] + '.npy'
        check_exist(folder_path)
        npy_file = os.path.join(folder_path, file_name)
        print('save to', npy_file)
        beat_curve = compute_beat_curve(audio_)
        np.save(npy_file, beat_curve, allow_pickle=False)
        break