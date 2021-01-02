
import ipdb
import librosa
import numpy as np
import glob
import os
from tqdm import tqdm
from argparse import ArgumentParser




def min_max_normalize(mel):
    margin = 0.9
    max_ = 19.0114
    min_ = 0.
    a = margin * (2.0 / (max_ - min_))
    b = margin * (-2.0 * min_ / (max_ - min_) - 1.0)
    data = mel * a + b
    return data


def compute_melgram(audio_path):
# mel-spectrogram parameters
    DURA = 29 #to make it 1366 frame 
    y, sr = librosa.load(audio_path)
    n_sample_fit = int(DURA*sr) 

    melgram = librosa.feature.melspectrogram
    ret = melgram(y=y[:n_sample_fit], sr=sr)[:,:1024]
    if ret.shape != (128, 1024):
        return None
    else:
        ret = np.log(ret*10000 +1)
        ret  = min_max_normalize(ret)
        return ret





def check_exist(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)



def get_mtg():
    for i in range(100):
        folder = '%02d' % i   
        audio_path = glob.glob('../baseline/audio/' + folder + '/*.mp3')
        with tqdm(total=len(audio_path), desc = folder) as pbar:
            for audio_ in audio_path:
                folder_path = os.path.join('../baseline/', 'vqvae', folder)
                file_name = audio_.split('/')[4][:-4]
                check_exist(folder_path)
                npy_file = os.path.join(folder_path, file_name)
                ret = compute_melgram(audio_)
                np.save(npy_file, ret, allow_pickle=False)
                pbar.update(1)
                    


def get_mer31k():
    PATH = '/mnt/md0/user_annahung/MER31k_old'
    audio_path = glob.glob(PATH +'/wav/*.wav')

    with tqdm(total=len(audio_path)) as pbar:
        for audio_ in audio_path:
            folder_path = os.path.join(PATH, 'mel_vqvae')
            file_name = audio_.split('/')[6][:-4]+ '.npy'
            check_exist(folder_path)
            npy_file = os.path.join(folder_path, file_name)
            ret = compute_melgram(audio_)
            if ret is not None:
                np.save(npy_file, ret, allow_pickle=False)
            pbar.update(1)


        
        
if __name__ == '__main__':
    get_mtg()






