from cifar10 import CIFAR
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
from Networks.dcgan import DCGAN
from utils import weight_initilizer
from torch.utils.tensorboard import SummaryWriter

def train(dataloader, device, num_epochs):
    Gen_Losses = []
    Dis_Losses = []
    lr = 0.0002
    beta1 = 0.5
    iter = 0
    z_dim, channels_img, f_g = 100, 3, 32
    img_shape = 32,32,3
    f_d = 32
    
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    
    algo = DCGAN(z_dim,channels_img,f_g,f_d,device,lr,beta1)
    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader):
            d_l, g_l = algo.train_step(data[0],iter ,writer_fake,writer_real)
            Gen_Losses.append(g_l)
            Dis_Losses.append(d_l)
            
            if i%50 ==0:
                print(f'Epoch:{epoch}  {i}/{len(dataloader)}  Gen_loss:{np.array(Gen_Losses).mean()}  Disc_loss:{np.array(Dis_Losses).mean()}')
                writer_fake.add_scalar('loss',np.array(Gen_Losses).mean(),i)
                writer_real.add_scalar('loss',np.array(Dis_Losses).mean(),i)
            
    return Gen_Losses, Dis_Losses
    

def main():
    path = '/Volumes/E/PapersWithCode/GAN/code/Data/cifar-10-batches-py'
    device = torch.device("mps")
    data = CIFAR(path)
    images, labels = data.get_train_data()
    images_tensor = torch.tensor(images,device=device,dtype=torch.float32)
    # labels_tensor = torch.tensor(labels,device=device)
    
    
    my_dataset = TensorDataset(images_tensor) 
    my_dataloader = DataLoader(my_dataset,shuffle=True, batch_size=32) 
    g,d = train(my_dataloader,device,5)
    
    
if __name__ == "__main__":
    main()