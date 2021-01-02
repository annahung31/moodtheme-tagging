import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim

import h5py 
import sys
import os
import json
import argparse
import numpy as np

import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter 

from trainer import Classifier_Trainer
from model import VectorQuantizedVAE
from GRU_model import RNNclassifier
from data_loader import get_mer31k_loader


def main(config, device):
    assert config.mode in {'TRAIN', 'TEST'},\
        'invalid mode: "{}" not in ["TRAIN", "TEST"]'.format(config.mode)


    if not os.path.exists(os.path.join(config.model_save_path,config.model_name)):
        os.makedirs(os.path.join(config.model_save_path,config.model_name))

    if not os.path.exists(os.path.join(config.roc_save_path,config.model_name)):
        os.makedirs(os.path.join(config.roc_save_path,config.model_name))
                

    model = VectorQuantizedVAE(1, config.hidden_size, config.k)
    # model.load_state_dict(torch.load(config.vqvae_path))
    # print(torch.load(config.vqvae_path).keys())
    model.load_state_dict(torch.load(config.vqvae_path)['vqvae_state_dict'],strict=False)  #yohua's way
    
    classifier = RNNclassifier(input_size=config.k, hidden_size=config.hidden_size, 
                                                    num_layers=2, num_classes=config.num_classes)


    model.to(device)
    classifier.to(device)




    if config.mode == 'TRAIN':
        data_loader = get_mer31k_loader(config.audio_path,
                                        config.batch_size,
                                        tr_val = 'train',
                                        )   
        valid_loader = get_mer31k_loader(config.audio_path,
                                        config.batch_size,
                                        tr_val = 'val',
                                        )   

        trainer = Classifier_Trainer(data_loader, valid_loader, model, classifier, config, device)

        trainer.train()

    elif config.mode == 'TEST':
        data_loader = get_mer31k_loader(config.audio_path,
                                        config.batch_size,
                                        tr_val = 'test',
                                        )   
        trainer = Classifier_Trainer(None, data_loader, model, classifier, config, device)
        trainer.test()




if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--dbtype', type=str, default='mtg')
    
    parser.add_argument('--vqvae_path', type=str, default='/mnt/md0/user_annahung/auto_tagging/mtg-jamendo-dataset/scripts/yohua_models/mtat/best_MTAT_classifaction.tar') 
    # parser.add_argument('--vqvae_path', type=str, default='/mnt/md0/user_annahung/auto_tagging/mtg-jamendo-dataset/scripts/VQVAE/models/CNN_ml/vqvae.pth')
    parser.add_argument('--num_classes', type=int, default=190)  #mtt = 50, mtg = 56
    parser.add_argument('--audio_path', type=str, default='/mnt/md0/user_annahung/MER31k_old')  #/mnt/md0/user_annahung/auto_tagging/mtt_dataset/  /mnt/md0/user_annahung/auto_tagging/mtg-jamendo-dataset/scripts/baseline
    parser.add_argument('--subset', type=str, default='mood')
    parser.add_argument('--split', type=int, default=0)
    
    parser.add_argument('--batch_size', type=int, default=12) 
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--mode', type=str, default='TRAIN')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--roc_save_path', type=str, default='./scores')
    parser.add_argument('--model_name', type=str, default='RNN_classifier')   #se_dilatedGRU
    # parser.add_argument('--roc_save_path', type=str, default='./scores')
    parser.add_argument('--hidden_size', type=int, default=256,help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=1024,help='number of latent vectors')
    parser.add_argument('--beta', type=float, default=1.0,help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')
    parser.add_argument('--lr', type=float, default=3e-5, metavar='LR', help='Learning rate.')
    parser.add_argument('--lr-decay', type=float, default=0.2, metavar='DC', help='Learning rate decay rate.')
    parser.add_argument('--weight-decay', type=float, default=0., metavar='WD', help='Weight decay.')
    parser.add_argument('--DROPOUT', type=float, default=0.5, metavar='DO', help='Dropout rate.')
    # parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='Momentum for SGD.')
    # parser.add_argument('--amplifying_ratio', type=int, default=16, metavar='A', help='Amplifying ratio of SE')
    config = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    
    print(config)
    main(config, device)
    print(config)