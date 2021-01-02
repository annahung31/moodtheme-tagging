# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import datetime
import tqdm
from sklearn import metrics
import pickle
import csv
import ipdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

class Classifier_Trainer(object):
    def __init__(self, data_loader, valid_loader, vqvae_model, classifier, config, device):
        self.config = config
        self.num_classes = config.num_classes
        self.beta = config.beta
        self.tag_list = self.get_tag_list(config)
        # Data loader
        self.data_loader = data_loader
        self.valid_loader = valid_loader
        self.device = device
        # Training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.log_step = config.log_step
        self.batch_size = config.batch_size

        #build model
        self.vqvae_model = vqvae_model
        self.classifier = classifier
        self.optimizer = torch.optim.Adam(list(self.vqvae_model.parameters())+list(self.classifier.parameters()), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')

        self.classifier_fn = os.path.join(config.model_save_path,config.model_name, 'classifier.pth')
        self.bce_loss = nn.BCELoss()
        self.roc_save_path = config.roc_save_path
        self.train_score_fn = os.path.join(self.roc_save_path,config.model_name, 'train_best_score_'+ datetime.datetime.now().strftime('%m-%d-%H_%M_%S') +'.txt')
        self.test_score_fn = os.path.join(self.roc_save_path,config.model_name, 'test_score_'+ datetime.datetime.now().strftime('%m-%d-%H_%M_%S') +'.txt')



    def train(self):
        start_t = time.time()
        train_loss = 0
        best_loss = 1000
        best_roc_auc = 0


        self.vqvae_model.zero_grad()
        self.classifier.zero_grad()

        for epoch in range(self.n_epochs):
            self.vqvae_model.train()
            self.classifier.train()
            self.optimizer.zero_grad()
            # train
            for ctr, (_, x, y) in enumerate(self.data_loader):
                x = self.to_var(x.unsqueeze(1))
                y = self.to_var(y)
                z_q_x_st, z_q_x, z_e_x = self.vqvae_model.encode(x)
                z_q_x_st = self.reshape_output(z_q_x_st)
                out_ = self.classifier(z_q_x_st)


                ###get loss
                # Vector quantization objective
                loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
                # Commitment objective
                loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
                #classification loss
                loss = self.bce_loss(out_,y)
                
                total_loss = loss + loss_vq + loss_commit
                total_loss.backward()
                self.optimizer.step()                    

                if ctr == 0:
                    print('-'*60, 'TRAIN', '-'* 60)
                if (ctr) % self.log_step == 0:
                    print("[%s] Epoch [%3d/%3d] Iter [%4d/%4d] classifier loss: %.10f vq_loss: %.6f Elapsed: %s" %
                            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            epoch+1, self.n_epochs, ctr, len(self.data_loader), loss.item() , loss_vq.item(), 
                            datetime.timedelta(seconds=time.time()-start_t)))

            # validation
            roc_auc, pr_auc, roc_auc_all, pr_auc_all = self._validation(start_t, epoch)

            # save model
            if roc_auc > best_roc_auc:
                print('best model: %4f' % roc_auc)
                best_roc_auc = roc_auc
                torch.save(self.classifier.state_dict(), self.classifier_fn)
                with open(self.train_score_fn, "w") as text_file:
                    for i in range(self.num_classes):
                        text_file.write('%-25s \t\t %.4f , %.4f' % (self.tag_list[i], roc_auc_all[i], pr_auc_all[i]))
                        text_file.write('\n')
                    text_file.write('epoch: %f' % (epoch))
                    text_file.write('average roc_auc: %.4f, pr_auc:  %.4f' % (roc_auc,pr_auc))
                


        
    def _validation(self, start_t, epoch):
        prd_array = []  # prediction
        gt_array = []   # ground truth
        ctr = 0
        self.vqvae_model.eval()
        self.classifier.eval()

        for ctr, (_, x, y) in enumerate(self.valid_loader):
            # variables to cuda
            x = self.to_var(x.unsqueeze(1))
            y = self.to_var(y)

            # predict
            z_q_x_st, z_q_x, z_e_x = self.vqvae_model.encode(x)
            z_q_x_st = self.reshape_output(z_q_x_st)
            out = self.classifier(z_q_x_st)

            # Vector quantization objective
            loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
            # Commitment objective
            loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
            #classification loss
            loss = self.bce_loss(out, y)
            total_loss = loss + loss_vq + loss_commit

            if ctr == 0:
                print('-'*62, 'VAL', '-'* 60)
            # print log
            if (ctr) % self.log_step == 0:
                print("VAL [%s] Epoch [%3d/%3d] Iter [%4d/%4d] classifier loss: %.10f vq_loss: %.6f Elapsed: %s" %
                            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            epoch+1, self.n_epochs, ctr, len(self.valid_loader), loss.item() , loss_vq.item(), 
                            datetime.timedelta(seconds=time.time()-start_t)))


            # append prediction
            out = out.detach().cpu()
            y = y.detach().cpu()
            for prd in out:
                prd_array.append(list(np.array(prd)))
            for gt in y:
                gt_array.append(list(np.array(gt)))
        ipdb.set_trace()
        roc_auc, pr_auc, roc_auc_all, pr_auc_all = self.get_auc(prd_array, gt_array)
        return roc_auc, pr_auc, roc_auc_all, pr_auc_all


    def test(self):
        start_t = time.time()
        self.vqvae_model.eval()
        self.classifier.eval()

        prd_array = []  # prediction
        gt_array = []   # ground truth

        with torch.no_grad():
            for ctr, (_, x, y) in enumerate(self.data_loader):
                x = self.to_var(x.unsqueeze(1))
                y = self.to_var(y)                

                z_q_x_st, z_q_x, z_e_x = vqvae_model.encode(x)
                z_q_x_st = self.reshape_output(z_q_x_st)
                out_ = self.classifier(z_q_x_st)


                # Vector quantization objective
                loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
                #classification loss
                loss = self.bce_loss(out_,y)

                if ctr == 0:
                    print('-'*61, 'TEST', '-'* 60)
                # print log
                if (ctr) % self.log_step == 0:
                    print("TEST [%s] Iter [%d/%d] test loss: %.6f loss_vq: %.6f Elapsed: %s" %
                            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            ctr, len(self.data_loader), loss.item(), loss_vq.item(), 
                            datetime.timedelta(seconds=time.time()-start_t)))

                # append prediction
                out = out.detach().cpu()
                y = y.detach().cpu()
                for prd in out:
                    prd_array.append(list(np.array(prd)))
                for gt in y:
                    gt_array.append(list(np.array(gt)))

        # get auc
        roc_auc, pr_auc, roc_auc_all, pr_auc_all = self.get_auc(prd_array, gt_array)

        # save aucs

        with open(self.test_score_fn, "w") as text_file:
            for i in range(self.num_classes):
                text_file.write('%-25s \t\t %.4f , %.4f' % (self.tag_list[i], roc_auc_all[i], pr_auc_all[i]))
                text_file.write('\n')
            text_file.write('epoch: %.f' % (epoch))
            text_file.write('\n')
            text_file.write('average roc_auc: %.4f, pr_auc:  %.4f' % (roc_auc,pr_auc))
            text_file.write('\n')
            text_file.write(str(self.config))
            



    def to_var(self, x):
        x = x.to(self.device)
        return x


    def get_tag_list(self, config):
        #mer31k
        path = os.path.join(config.audio_path, 'tagsName.npy')       
        tag_list = np.load(path)
        return tag_list




    def reshape_output(self, out):
        batch_size, feature_dim, feature_num, length = out.shape
        out = out.view(batch_size,feature_dim*feature_num, length)
        out = out.permute(0,2,1)
        return out


    def get_auc(self, prd_array, gt_array):
        
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






class VQVAE_Trainer(object):
    def __init__(self, data_loader, test_loader, vqvae_model, config, device):
        self.config = config
        self.beta = config.beta
        # Data loader
        self.data_loader = data_loader
        self.test_loader = test_loader
        self.device = device
        # Training settings
        self.n_epochs = config.n_epochs
        self.lr = config.lr
        self.log_step = config.log_step
        self.batch_size = config.batch_size
        # Build model
        self.vqvae_model = vqvae_model
        self.optimizer = torch.optim.Adam(self.vqvae_model.parameters(), lr=self.lr)
        self.vqvae_model_fn = os.path.join(config.model_save_path,config.model_name, 'vqvae.pth')
        self.image_fn = os.path.join(config.model_save_path, config.model_name)
        
    def to_var(self, x):
        x = x.unsqueeze(1)
        x = x.to(self.device)
        return x


    def train(self):
        start_t = time.time()
        best_loss = 10000

        for epoch in range(self.n_epochs):
            # train
            self.vqvae_model.train()

            for ctr, (_, x, _) in enumerate(self.data_loader):
                x = self.to_var(x)
                self.optimizer.zero_grad()

                # training
                x_tilde, z_e_x, z_q_x = self.vqvae_model(x)

                # Reconstruction loss
                loss_recons = F.mse_loss(x_tilde, x)

                # Vector quantization objective
                loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
                # Commitment objective
                loss_commit = F.mse_loss(z_e_x, z_q_x.detach())

                train_loss = loss_recons + loss_vq + self.beta * loss_commit
                train_loss.backward()
                self.optimizer.step()    

                #print loss
                if (ctr) % self.log_step == 0:
                    print("[%s] Epoch [%3d/%3d] Iter [%4d/%4d] train loss: %.4f Elapsed: %s" %
                            (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            epoch+1, self.n_epochs, ctr, len(self.data_loader), train_loss.item(),
                            datetime.timedelta(seconds=time.time()-start_t)))

            self.generate_picture(epoch, x, x_tilde)

            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(self.vqvae_model.state_dict(), self.vqvae_model_fn)


            self.test()


    def generate_picture(self, epoch, label_spec ,predict_spec, valid=False):
        label_spec = label_spec.squeeze(1)
        predict_spec = predict_spec.squeeze(1)

        real = label_spec.data.cpu().numpy()[0,:,:]
        fake = predict_spec.data.cpu().numpy()[0,:,:]
        stack_spec = np.hstack((real, fake))

        fig = plt.figure(figsize=(20,3))
        plt.imshow(stack_spec,aspect='auto')
        if valid:
            plt.savefig( self.image_fn + "/Valid_{}_sample.png".format(epoch))
        else:
            plt.savefig( self.image_fn + "/{}_sample.png".format(epoch))

        plt.close()


    def load(self, filename):
        S = torch.load(filename)
        self.model.load_state_dict(S)

    def save(self, filename):
        model = self.model.state_dict()
        torch.save({'model': model}, filename)



    def test(self):
        start_t = time.time()
        epoch = 0
        dataset_size = len(self.test_loader)
        
        prd_array = []  # prediction
        gt_array = []   # ground truth

        self.vqvae_model.load_state_dict(torch.load(self.vqvae_model_fn))

        with torch.no_grad():
            self.vqvae_model.eval()
            for ctr, (_, x, _) in enumerate(self.test_loader):
                x = self.to_var(x)
                x_tilde, z_e_x, z_q_x = self.vqvae_model(x)
                
                #loss
                loss_recons += F.mse_loss(x_tilde, x)
                loss_vq += F.mse_loss(z_q_x, z_e_x)
                loss_commit += F.mse_loss(z_e_x, z_q_x)


            loss_recons /= dataset_size
            loss_vq /= dataset_size     
            loss_commit /= dataset_size
            print("Average loss: loss_recons: %.4f  loss_vq: %.4f  loss_commit: %.4f" %
                            (loss_recons.item(), loss_vq.item(), loss_commit.item()))

            self.generate_picture(0, x, x_tilde, valid=True)








