import sys
import os
import pickle
import numpy as np

class CIFAR:
    def __init__(self,path):
        
        self.train_filenames = [path+'/'+f'data_batch_{i}' for i in range(1,6)]
        self.meta_file = path+'/'+'batches.meta'
        self.test_filename = path+'/'+'test_batch'
        
        self.train_images = np.zeros((50000,3,32,32), dtype='uint8')
        self.train_labels = np.zeros((50000,), dtype='int32')
        
        
        
    def get_train_data(self):
        
        for i,file in enumerate(self.train_filenames):
            with open(file, 'rb') as fo:
                dict = pickle.load(fo, encoding='bytes')
            
            
            self.train_images[i*10000:10000*(i+1),:,:,:] =  dict[b'data'].reshape(10000,3,32,32)
            self.train_labels[i*10000:10000*(i+1)] = dict[b'labels']
            
        
        self.train_images = (self.train_images -  127.5) / 127.5
        return self.train_images, self.train_labels
        
        
        
        
        