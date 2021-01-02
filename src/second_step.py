import warnings
warnings.filterwarnings("ignore")
import ipdb
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torch.utils.data as utils
import h5py 
import sys
import os
import json
from model import VectorQuantizedVAE
# from model_zoo.VQVAE import VectorQuantizedVAE

from GRU_model import RNNclassifier, CNN
# from model_zoo.awd_GRU import RNNclassifier as awdRNN_classifier

import matplotlib.pyplot as plt
import argparse
from tensorboardX import SummaryWriter 

from sklearn.metrics import roc_auc_score
from sklearn import metrics
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from torch.utils.data.dataloader import default_collate

from data_loader import get_mer31k_loader, get_mtg_loader


def get_tag_list(audio_path, type_):
    #mer31k
    if type_ == 'mer31k':
        path = os.path.join(audio_path, 'tagsName.npy')       
        
    elif type_ == 'mtg':
        split = 0
        subset = 'mood'
        path = os.path.join(audio_path, 'split', subset, 'split-'+str(split), 'tag_list.npy') 
    tag_list = np.load(path)
    return tag_list


def get_auc(prd_array, gt_array):
    
    prd_array = np.array(prd_array)
    gt_array = np.array(gt_array)

    roc_aucs = metrics.roc_auc_score(gt_array, prd_array, average='macro')
    pr_aucs = metrics.average_precision_score(gt_array, prd_array, average='macro')

    print('roc_auc: %.4f' % roc_aucs)
    print('pr_auc: %.4f' % pr_aucs)

    roc_auc_all = metrics.roc_auc_score(gt_array, prd_array, average=None)
    pr_auc_all = metrics.average_precision_score(gt_array, prd_array, average=None)

    # for i in range(self.num_classes):
    #     print('%-35s \t\t %.4f , %.4f' % (self.tag_list[i], roc_auc_all[i], pr_auc_all[i]))
    return roc_aucs, pr_aucs, roc_auc_all, pr_auc_all




def train(vqvae_model, classifeir_model, epoch, train_loader, optimizer, args, writer):
    vqvae_model.train()
    classifeir_model.train()
    train_loss = 0
    best_roc_auc = 0
    bce_loss = nn.BCELoss()
    prd_array = []  # prediction
    gt_array = []   # ground truth
    
    for batch_idx, (_,data, label) in enumerate(train_loader):    
        B = data.shape[0]
        optimizer.zero_grad()
        
        # data = torch.FloatTensor(data)    
        label = label.cuda()
        data = data.unsqueeze(1).cuda()
        

        z_q_x_st, z_q_x, z_e_x = vqvae_model.encode(data)
        # print("z_q_x_st",z_q_x_st.shape)
        B, feature_dim, feature_num, length= z_q_x_st.shape
        z_q_x_st = z_q_x_st.view(B,feature_dim*feature_num, length)
        # print("reshape z_q_x_st",z_q_x_st.shape)
        z_q_x_st = z_q_x_st.permute(0,2,1) # batch, feature, seq ->batch, seq, feature

        # exit()

        # Vector quantization objective
        loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
        # Commitment objective
        loss_commit = F.mse_loss(z_e_x, z_q_x.detach())


        out_ = classifeir_model(z_q_x_st)        
        loss = bce_loss(out_,label)

        total_loss = loss + loss_vq + loss_commit
        total_loss.backward()
        optimizer.step()    

        '''
        #auc socre
        out_np = out_.view(B*190).cpu().data.numpy()
        label_np =  label.view(B*190).cpu().data.numpy().astype(int)
        # out_np = out_.cpu().data.numpy()
        # label_np =  label.cpu().data.numpy().astype(int)

        auc = roc_auc_score(label_np, out_np) 
        # exit()
        # fpr,tpr,thresholds=metrics.roc_curve(label_np,out_np,pos_label=1)
        # auc = metrics.auc(fpr,tpr)
        '''

        # my aucs 
        out_ = out_.detach().cpu()
        label = label.detach().cpu()
        for prd in out_:
            prd_array.append(list(np.array(prd)))
        for gt in label:
            gt_array.append(list(np.array(gt)))

        '''        
        # Logs
        writer.add_scalar('loss/train/loss', loss.item()/len(data), epoch*len(train_loader)+batch_idx)
        writer.add_scalar('loss/train/auc', auc, epoch*len(train_loader)+batch_idx)
        '''
         
        if batch_idx % 20 == 0:
            print ('Train Epoch: {} [{}/{} ({:.0f}%)]\t loss: {:.6f}, vq_loss: {:.6f}  '.format(epoch+1,
                 batch_idx * args.batch_size, len(train_loader.dataset), 100. * batch_idx/len(train_loader), 
                 loss.item()/len(data), loss_vq.item()/len(data)  ))

    roc_auc, pr_auc, roc_auc_all, pr_auc_all = get_auc(prd_array, gt_array)  


    

def test(vqvae_model, classifeir_model, epoch, test_loader, scheduler, args, writer, image_path):
    vqvae_model.eval()
    classifeir_model.eval()
    train_loss = 0
    bce_loss = nn.BCELoss()

    with torch.no_grad():
        count=0
        loss = 0.
        '''
        auc=0.
        '''

        prd_array = []
        gt_array = []
        for batch_idx, (_,data, label) in enumerate(test_loader):

            B = data.shape[0]

            label = label.cuda()
            data = data.unsqueeze(1).cuda()
            
            z_q_x_st, z_q_x, z_e_x = vqvae_model.encode(data)
            B, feature_dim, feature_num, length= z_q_x_st.shape
            z_q_x_st = z_q_x_st.view(B,feature_dim*feature_num, length)
            z_q_x_st = z_q_x_st.permute(0,2,1) # batch, feature, seq ->batch, seq, feature
            out_ = classifeir_model(z_q_x_st)

            loss += bce_loss(out_,label)

            '''
            out_np = out_.view(B*190).cpu().data.numpy()
            label_np =  label.view(B*190).cpu().data.numpy().astype(int)
            auc += roc_auc_score(label_np, out_np)

            stack_label.append(label.squeeze().cpu().data.numpy().astype(int)) 
            stack_result.append(out_.squeeze().cpu().data.numpy())        
            '''
            # my aucs 
            out_ = out_.detach().cpu()
            label = label.detach().cpu()
            for prd in out_:
                prd_array.append(list(np.array(prd)))
            for gt in label:
                gt_array.append(list(np.array(gt)))



            count+=1
        scheduler.step(loss)
        '''
        loss /= len(test_loader)
        auc /= len(test_loader)
        '''
        roc_auc, pr_auc, roc_auc_all, pr_auc_all = get_auc(prd_array, gt_array)  


        

        
        # ipdb.set_trace()
        # stack_label = np.stack(stack_label)
        # stack_result = np.stack(stack_result)
        # print(stack_label.shape)
        # print(stack_result.shape)
        # print(np.unique(stack_label))
        
        # new_auc = roc_auc_score(stack_label,stack_result)

        # fpr,tpr,thresholds=metrics.roc_curve(stack_label,stack_result)
        # new_auc = metrics.auc(fpr,tpr)

    # Logs
    print("-------------------------------------------------------")
    print("-------------------------------------------------------")
    print("---------Test auc:{:.10f}-------------".format(roc_auc))
    # print("---------Test epoch:{} new auc:{}-------------".format(epoch, new_auc))
    print("-------------------------------------------------------")
    print("-------------------------------------------------------")

    '''
    writer.add_scalar('loss/test/loss', loss.item(),  epoch*len(test_loader)+batch_idx)
    writer.add_scalar('loss/test/auc', auc,  epoch*len(test_loader)+batch_idx)
    '''

    return loss.item(), roc_auc, pr_auc, roc_auc_all, pr_auc_all




def main(args, log_path, model_path, image_path, score_path, comment):    
    # torch.autograd.set_detect_anomaly(True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    """ Define writer"""
    writer = SummaryWriter(log_path, comment=comment)

    """ Define Loader """
    if args.num_classes == 190:
        train_loader = get_mer31k_loader(args.audio_path,
                                    args.batch_size,
                                    tr_val = 'train',
                                    )   
        test_loader = get_mer31k_loader(args.audio_path,
                                    args.batch_size,
                                    tr_val = 'test',
                                    )  
        tag_list = get_tag_list(args.audio_path, 'mer31k')

    elif args.num_classes == 56:
        subset = 'mood'
        split = '0'
        train_loader = get_mtg_loader(args.audio_path,
                                        subset,
                                        args.batch_size,
                                        tr_val = 'train',
                                        split = split)
        test_loader = get_mtg_loader(args.audio_path,
                                        subset,
                                        args.batch_size,
                                        tr_val='test',
                                        split = split)        
        tag_list = get_tag_list(args.audio_path, 'mtg')



    """ Define model Optimizer"""
    VQ_VAE_model = VectorQuantizedVAE(1, args.hidden_size, args.k).cuda()
    VQ_VAE_model.load_state_dict(torch.load(args.vqvae_path)['vqvae_state_dict'],strict=False)
    # print(VQ_VAE_model)

    rnn_classifier =  RNNclassifier(input_size=args.k, hidden_size=args.hidden_size, num_layers=args.gru_layers, num_classes=args.num_classes).cuda()
    # CNN_classifier = CNN(num_class = args.num_classes, margin = 4, device = device).cuda()

    optimizer = optim.Adam(list(VQ_VAE_model.parameters())+list(rnn_classifier.parameters()), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    VQ_VAE_model.zero_grad()
    rnn_classifier.zero_grad()
    optimizer.zero_grad()



    print ('start training')
    best_loss = 1000
    best_roc_auc = 0
    for epoch in range(args.num_epochs):
        print(args.experiment_name)
        loss = train(VQ_VAE_model, rnn_classifier, epoch, train_loader, optimizer, args, writer)
        if epoch   % args.test_freq == 0:
            test_loss, roc_auc, pr_auc, roc_auc_all, pr_auc_all = test(VQ_VAE_model,rnn_classifier, epoch, test_loader, scheduler, args, writer, image_path)
            if test_loss < best_loss:     
                best_loss = test_loss
                torch.save({'epoch': epoch + 1, 'classifier_state_dict': rnn_classifier.state_dict(), 'vqvae_state_dict':VQ_VAE_model.state_dict() }, os.path.join(model_path, 'best_loss_checkpoint-{}.tar'.format(str(epoch + 1 ))))
                
            if roc_auc > best_roc_auc:
                print('best model: %4f' % roc_auc)
                best_roc_auc = roc_auc
                torch.save({'epoch': epoch + 1, 'classifier_state_dict': rnn_classifier.state_dict(), 'vqvae_state_dict':VQ_VAE_model.state_dict() }, os.path.join(model_path, 'best_auc_checkpoint-{}.tar'.format(str(epoch + 1 ))))
                with open(os.path.join(score_path, 'best_auc_ep_{}.txt'.format(str(epoch + 1 ))), "w") as text_file:
                    for i in range(args.num_classes):
                        text_file.write('%-25s \t\t %.4f , %.4f' % (tag_list[i], roc_auc_all[i], pr_auc_all[i]))
                        text_file.write('\n')
                    text_file.write('epoch: %f' % (epoch))
                    text_file.write('average roc_auc: %.4f, pr_auc:  %.4f' % (roc_auc,pr_auc))
                
        # exit()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VQ-VAE')
    # General
   
    parser.add_argument('--gpu', type=str, default= '1', help='gpu')


    parser.add_argument('--comment', type=str, default="_with_dilation_encoder_4-512_checkpoint35_correct_metric_gru_3linear",
        help='string after writer comment')

    parser.add_argument('--num_classes', type=int, default=56,
        help='num_classes')



    parser.add_argument('--gru_layers', '-g', type=int, default=2,
        help='gru_layers')

    # Latent space
    parser.add_argument('--hidden_size', type=int, default=256,
        help='size of the latent vectors (default: 256)')
    parser.add_argument('--k', type=int, default=1024,
        help='number of latent vectors (default: 512)')

   # Optimization
    parser.add_argument('--batch_size', type=int, default=12,
        help='batch size (default: 128)')
    parser.add_argument('--num-epochs', type=int, default=100,
        help='number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=2e-4,
        help='learning rate for Adam optimizer (default: 2e-4)')
    parser.add_argument('--beta', type=float, default=1.0,
        help='contribution of commitment loss, between 0.1 and 2.0 (default: 1.0)')

    # Miscellaneous
    
    parser.add_argument('--experiment_name', '-n', type=str, default='secondstep',
        help='experiment_name')
    
    parser.add_argument('--test_freq', type=int, default=1,
        help='test Frequency')
    # parser.add_argument('--vqvae_path', type=str, default='min_max_norm_mel_spectrum/models/K:1024_hiidenSize:256_experimentName:min_max_norm_mel_spectrum_dilation_encoder_1-512_wholeMSD/checkpoint-2.tar',
        # help='test Frequency')
    
    parser.add_argument('--vqvae_path', type=str, default='/mnt/md0/user_annahung/auto_tagging/mtg-jamendo-dataset/scripts/yohua_models/mtat/best_MTAT_classifaction.tar',
        help='test Frequency')

    parser.add_argument('--audio_path', type=str, default='/mnt/md0/user_annahung/auto_tagging/mtg-jamendo-dataset/scripts/baseline',
        help='audio_path')   #/mnt/md0/user_annahung/auto_tagging/mtg-jamendo-dataset/scripts/baseline

    
    args = parser.parse_args()


    comment = "K:{}_hiidenSize:{}_experimentName:{}".format(args.k, args.hidden_size, args.experiment_name + args.comment )
    
    if not os.path.exists(os.path.join('experiments'):
        os.makedirs(os.path.join('experiments',args.experiment_name)

    
    log_path = os.path.join('experiments', 'logs', args.experiment_name, comment)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    model_path = os.path.join('experiments', 'models',args.experiment_name, comment)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    image_path = os.path.join('experiments', 'images',args.experiment_name, comment)
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    score_path = os.path.join('experiments', 'scores',args.experiment_name, comment)
    if not os.path.exists(score_path):
        os.makedirs(score_path)


    print("log_path",log_path)
    print("model_path",model_path)
    print("image_path",image_path)
    print("score_path",image_path)
    main(args, log_path, model_path, image_path, score_path, comment)





