import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import math
from lsoftmax import LSoftmaxLinear

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class FCclassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.load_embedding_matrix()
        
        # self.GRU_layer = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        self.GRU_layer = nn.GRU(input_size, hidden_size, num_layers,bidirectional=True, batch_first=True, dropout=0.1)
        

        self.second_GRU_layer = nn.GRU(hidden_size*2, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=0.1)


        self.third_GRU_layer = nn.GRU(hidden_size*2, hidden_size, num_layers,bidirectional=True, batch_first=True, dropout=0.1)

        self.first_linear = nn.Linear(2*num_layers*hidden_size, 512)
        self.second_linear = nn.Linear(512,256)
        self.third_linear = nn.Linear(256,256)

        self.first_layer_norm = nn.LayerNorm((512,512))
        self.second_layer_norm = nn.LayerNorm((512,512))
        self.layer_norm = nn.LayerNorm((4,256))


        self.classifier_layer = nn.Linear(256,50)
        
        

        self.gelu = GELU()

        self.relu_layer = nn.ReLU(0.05)
        self.sigmoid = nn.Sigmoid()

        
        self.linear_layers = nn.Sequential(
            nn.Linear(1024*512, 512),
            
            # nn.Linear(2*num_layers*hidden_size, 512),
            
            nn.Dropout(p=0.05),
            nn.ReLU(0.1),
            
            ## nn.Linear(512,256),      
            ## nn.Dropout(p=0.1),    
            ## nn.ReLU(0.1),
            
            nn.Linear(512,256),    
            nn.Dropout(p=0.05),
            nn.ReLU(0.1),
            nn.Linear(256,self.num_classes),
            nn.Sigmoid()
        )
        

    def load_embedding_matrix(self):
        self.embedding = nn.Embedding(num_embeddings=1024, embedding_dim=256)
        
    def forward(self, latent_vector):
        B = latent_vector.shape[0]



        '''
        # latent_vector = self.embedding(input_)
        # print("latent_vector",latent_vector.shape)
        output , h_n  = self.GRU_layer(latent_vector) # input:batch, seq, feature
        # print("output ",output .shape)
        # print("h_n",h_n.shape)
        output = self.first_layer_norm(output)
        # output = self.gelu(output)

        output , h_n  = self.second_GRU_layer(output) # input:batch, seq, feature
        # print("output ",output.shape)

        # output = self.second_layer_norm(output)
        # # output = self.gelu(output)
        

        # output, h_n = self.third_GRU_layer(output)
        
        # print("output ",output .shape)
        # print("h_n",h_n.shape)
        



        h_n = h_n.permute(1,0,2)
        h_n = self.layer_norm(h_n)
        h_n = h_n.contiguous().view(B,-1)
        

        # h_n = h_n.permute(1,0,2).contiguous().view(B,-1)
        # print("h_n",h_n.shape)
        
        
        # after_first_linear = self.relu_layer(self.first_linear(h_n))

        # after_first_linear = F.dropout(after_first_linear, p=0.2, training=self.training)

        # after_second_linear = self.relu_layer(self.second_linear(after_first_linear))
        # after_second_linear = F.dropout(after_second_linear, p=0.2, training=self.training)

        # after_third_linear = self.relu_layer(self.third_linear(after_second_linear))
        # third_linear = F.dropout(after_third_linear, p=0.2, training=self.training)

    

        # self.output =  self.sigmoid(self.classifier_layer(after_third_linear))
        
        self.output = self.linear_layers(h_n)
        '''
        latent_vector = latent_vector.contiguous().view(B, -1)
        self.output = self.linear_layers(latent_vector)        
        return self.output







class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, k_size=3, std = 1, pad = 1, poo_x=2, poo_y=4):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size = k_size, stride = std, padding = pad)
        self.bn = nn.BatchNorm2d(out_planes)
        self.mp = nn.MaxPool2d((poo_x, poo_y))

    def forward(self, x):
        x = self.mp(nn.ELU()(self.bn(self.conv(x))))
        return x



class CNN(nn.Module):
    '''
    CNN with feature extractor and classifier
    '''
    def __init__(self, num_class, margin, device):
        super(CNN, self).__init__()
        # init bn
        self.device = device
        self.bn_init = nn.BatchNorm2d(1)
        self.c_ = 1
        self.margin = margin
        self.num_classes = num_class
        self.feature_extractor = nn.Sequential(
            ConvBlock(  1, int(64*self.c_), 3, 1, 1, 2, 4),
            ConvBlock(int(64*self.c_),int(128*self.c_), 3, 1, 1, 2, 4),
            ConvBlock(int(128*self.c_),int(256*self.c_), 3, 1, 1, 2, 4),
            ConvBlock(int(256*self.c_),int(512*self.c_), 3, 1, 1, 3, 5),
            ConvBlock(int(512*self.c_), 256, 3, 1, 1, 4, 4)
        )


        self.dense = nn.Sequential(
            nn.Linear(1280, self.num_classes),
            nn.BatchNorm1d(self.num_classes)
        )

        # self.dense = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.5)
        # self.acti_layer = nn.Softmax(dim=1)
        self.acti_layer = nn.Sigmoid()
    
    def extractor(self, x):
        x = x.unsqueeze(1)
        x = self.bn_init(x)
        x = self.feature_extractor(x)
        return x

    def classifier(self,x):
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        assert x.shape[-1] == 1280 
        predict = self.acti_layer(self.dense(x))
        
        return predict

    def forward(self, x):
        x = self.extractor(x)
        predict = self.classifier(x)
        return predict






class RNNclassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.load_embedding_matrix()
        
        # self.GRU_layer = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)

        self.GRU_layer = nn.GRU(input_size, hidden_size, num_layers,bidirectional=True, batch_first=True, dropout=0.3)
        

        self.second_GRU_layer = nn.GRU(hidden_size*2, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=0.3)


        self.third_GRU_layer = nn.GRU(hidden_size*2, hidden_size, num_layers,bidirectional=True, batch_first=True, dropout=0.2)

        self.first_linear = nn.Linear(2*num_layers*hidden_size, 512)
        self.second_linear = nn.Linear(512,256)
        self.third_linear = nn.Linear(256,256)

        self.first_layer_norm = nn.LayerNorm((512,512))
        self.second_layer_norm = nn.LayerNorm((512,512))
        self.layer_norm = nn.LayerNorm((2*num_layers,256))


        self.classifier_layer = nn.Linear(256,50)
        
        

        self.gelu = GELU()

        self.relu_layer = nn.ReLU(0.05)
        self.sigmoid = nn.Sigmoid()

        
        self.linear_layers = nn.Sequential(
            nn.Linear(1024, 512),
            
            # nn.Linear(2*num_layers*hidden_size, 512),
            
            nn.Dropout(p=0.05),
            nn.ReLU(0.1),
            
            ## nn.Linear(512,256),      
            ## nn.Dropout(p=0.1),    
            ## nn.ReLU(0.1),
            
            nn.Linear(512,256),    
            nn.Dropout(p=0.05),
            nn.ReLU(0.1),
            nn.Linear(256,self.num_classes),
            nn.Sigmoid()
        )
        

    def load_embedding_matrix(self):
        self.embedding = nn.Embedding(num_embeddings=1024, embedding_dim=256)
        
    def forward(self, latent_vector):
        B = latent_vector.shape[0]



        
        # latent_vector = self.embedding(input_)
        output , h_n  = self.GRU_layer(latent_vector) # input:batch, seq, feature
        output = self.first_layer_norm(output)


        output , h_n  = self.second_GRU_layer(output) # input:batch, seq, feature

        # output = self.second_layer_norm(output)
        # # output = self.gelu(output)
        

        # output, h_n = self.third_GRU_layer(output)
        
        


        h_n = h_n.permute(1,0,2)
        h_n = self.layer_norm(h_n)
        h_n = h_n.contiguous().view(B,-1)
        

        # h_n = h_n.permute(1,0,2).contiguous().view(B,-1)
        # print("h_n",h_n.shape)
        
        
        # after_first_linear = self.relu_layer(self.first_linear(h_n))

        # after_first_linear = F.dropout(after_first_linear, p=0.2, training=self.training)

        # after_second_linear = self.relu_layer(self.second_linear(after_first_linear))
        # after_second_linear = F.dropout(after_second_linear, p=0.2, training=self.training)

        # after_third_linear = self.relu_layer(self.third_linear(after_second_linear))
        # third_linear = F.dropout(after_third_linear, p=0.2, training=self.training)

    

        # self.output =  self.sigmoid(self.classifier_layer(after_third_linear))
        self.output = self.linear_layers(h_n)
        return self.output