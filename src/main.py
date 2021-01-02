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

from trainer import VQVAE_Trainer
from model import VectorQuantizedVAE
from data_loader import get_mer31k_loader


def main(config, device):
    assert config.mode in {'TRAIN', 'TEST'},\
        'invalid mode: "{}" not in ["TRAIN", "TEST"]'.format(config.mode)


    if not os.path.exists(os.path.join(config.model_save_path,config.model_name)):
        os.makedirs(os.path.join(config.model_save_path,config.model_name))

        

    VQ_VAE_model = VectorQuantizedVAE(1, args.hidden_size, args.k)
    VQ_VAE_model.load_state_dict(torch.load(args.vqvae_path)['vqvae_state_dict'],strict=False)
    VQ_VAE_model.to(device)
    rnn_classifier =  RNNclassifier(input_size=args.k, hidden_size=args.hidden_size, num_layers=args.gru_layers, num_classes=args.num_classes).cuda()
    rnn_classifier.to(device)

    if config.mode == 'TRAIN':
        data_loader = get_mer31k_loader(config.audio_path,
                                        config.batch_size,
                                        tr_val = 'train',
                                        )
        # valid_loader = get_mer31k_loader(config.audio_path,
        #                                 config.batch_size,
        #                                 tr_val = 'val',
        #                                 )
        test_loader = get_mer31k_loader(config.audio_path,
                                        config.batch_size,
                                        tr_val = 'test',
                                        )        
        trainer = VQVAE_Trainer(data_loader, test_loader, VQ_VAE_model, rnn_classifier, config, device)

        trainer.train()

    elif config.mode == 'TEST':
        data_loader = get_mer31k_loader(config.audio_path,
                                        config.batch_size,
                                        tr_val = 'test',
                                        )
        trainer = VQVAE_Trainer(None, data_loader, VQ_VAE_model, rnn_classifier, config, device)

        trainer.test()




if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--gpu', type=str, default='0')
    # parser.add_argument('--dbtype', type=str, default='mtg')
    # parser.add_argument('--num_classes', type=int, default=190)  #mtt = 50, mtg = 56
    parser.add_argument('--audio_path', type=str, default='/mnt/md0/user_annahung/auto_tagging/mtg-jamendo-dataset/scripts/baseline')  #/mnt/md0/user_annahung/auto_tagging/mtt_dataset/  /mnt/md0/user_annahung/auto_tagging/mtg-jamendo-dataset/scripts/baseline
    parser.add_argument('--subset', type=str, default='mood')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=12) 
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--log_step', type=int, default=100)
    parser.add_argument('--mode', type=str, default='TRAIN')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--model_name', type=str, default='CNN_ml')   #se_dilatedGRU
    # parser.add_argument('--roc_save_path', type=str, default='./scores')
    parser.add_argument('--hidden_size', type=int, default=256,help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=1024,help='number of latent vectors')
    parser.add_argument('--beta', type=float, default=1.0,help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR', help='Learning rate.')
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