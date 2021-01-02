import os
import argparse
import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

from torch.utils.data.dataloader import default_collate

class MSD_Dataset(Dataset):
    def __init__(self, Data_path):
        ##############################################
        ### Initialize paths, transforms, and so on
        mean = np.load("np_mean_std/mean.npy").reshape(128,1)
        self.mean = np.repeat(mean, 1024, axis=1)
        
        std = np.load("np_mean_std/std.npy").reshape(128,1)

        self.std = np.repeat(std, 1024, axis=1)

        self.max = 19.0114
        self.min = 0.0
        self.min_max_range_normalize(0.9)
        self.Data_path = os.listdir(Data_path)
        print(len(self.Data_path))
    
    
    def min_max_range_normalize(self, margin):
        self.a = margin * (2.0 / (self.max - self.min))
        self.b = margin * (-2.0 * self.min / (self.max - self.min) - 1.0)
    
    def min_max_normalize(self, data):
        
        data = data *self.a + self.b
        return data
    
    def normalize(self, data):
        data = np.divide((data-self.mean), self.std)
        return data
    
    def __getitem__(self, index):
        
        try:
            single_path = self.Data_path[index]
            ipdb.set_trace()
            mel_spec = np.load(os.path.join('NewData/train',single_path), allow_pickle=True)[:,:1024]
        
            assert mel_spec.shape ==(128,1024)
        
        except AssertionError:
            return None
            
        except OSError:
            return None
#             print(single_path)
#             return None

        mel_spec = np.log(mel_spec*10000 +1)
        mel_spec  = self.min_max_normalize(mel_spec)

        return mel_spec

    def __len__(self):
        ##############################################
        ### Indicate the total size of the dataset
        ##############################################
        return len(self.Data_path)
    
def my_collate(batch):
    batch = list(filter(lambda x : x is not None, batch))        

    batch = torch.FloatTensor(batch)
    # return default_collate(batch)
    return batch

def Process_Data(args):

    msd_D = MSD_Dataset('NewData/train')    
    train_loader = DataLoader(dataset=msd_D,
                          batch_size=args.batch_size, 
                          shuffle=True,
                          num_workers=24,
                         collate_fn=my_collate)
            # break 

    return train_loader, None





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VQ-VAE')
    # General


    # parser.add_argument('--k', type=int, default=1024,
    #     help='number of latent vectors (default: 512)')

   # Optimization
    parser.add_argument('--batch_size', type=int, default=12,
        help='batch size (default: 128)')

    args = parser.parse_args()


    Process_Data(args)