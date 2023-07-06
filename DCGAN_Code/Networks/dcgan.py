import torch
import torch.nn as nn
import torchvision

class Generator(nn.Module):
    def __init__(self,z_dim, channels_img, f_g):
        super().__init__()
        
        self.net = nn.Sequential(
            self._block(z_dim, f_g*16, 4, 1, 0), #4X4
            self._block(f_g*16, f_g*8, 4, 2, 1), #8X8
            self._block(f_g*8, f_g*4, 4, 2, 1), #16X16
            # self._block(f_g*4, f_g*2, 4, 2, 1),
            nn.ConvTranspose2d(f_g*4, channels_img, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.net(x)
    
    

class Discriminator(nn.Module):
    def __init__(self,n_c, f_d):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(n_c, f_d*4, 4, 2, 0,bias=False),
            nn.LeakyReLU(0.2),
            self._block(f_d*4,f_d*8, 4, 2, 0),
            self._block(f_d*8,f_d*16, 4, 2, 0),
            nn.Conv2d(f_d*16, 1, 2, 2, 0,bias=False),
            nn.Sigmoid()
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
        
    def forward(self, x):
        return self.net(x)
    
    
class DCGAN:
    def __init__(self, z_dim, channels_img, f_g, f_d, device,lr, beta1):
        self.device = device
        self.z_in = z_dim
        self.netG = Generator(z_dim, channels_img, f_g).to(device)
        self.netD = Discriminator(channels_img, f_d).to(device)
        self.optG = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=[beta1, 0.999])
        self.optD = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=[beta1, 0.999])
        self.loss = nn.BCELoss()
    
    def train_step(self, x_batch,iter,writer_fake,writer_real):
        self.optD.zero_grad()
        d_out_r = self.netD(x_batch).view(-1)
        real_label = torch.ones((x_batch.shape[0],), device=self.device)
        d_loss_r = self.loss(d_out_r, real_label)
        
        z = torch.rand(size=(x_batch.shape[0],self.z_in,1,1), device=self.device)
        g_out = self.netG(z)
        d_out_f = self.netD(g_out).view(-1)
        fake_label = torch.zeros((x_batch.shape[0],), device=self.device)
        d_loss_f = self.loss(d_out_f, fake_label)
        
        d_loss = d_loss_r + d_loss_f
        d_loss.backward(retain_graph=True)
        self.optD.step()
        
        self.optG.zero_grad()
        g_out_r = self.netD(g_out).view(-1)
        g_loss = self.loss(g_out_r, real_label)
        g_loss.backward()
        self.optG.step()
        
        if iter%50 == 0:
            with torch.no_grad():
                fake = self.netG(z)
                # take out (up to) 32 examples
                img_grid_real = torchvision.utils.make_grid(x_batch[:32], normalize=True)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)

                writer_real.add_image("Real", img_grid_real, global_step=iter)
                writer_fake.add_image("Fake", img_grid_fake, global_step=iter)

        return d_loss.item(),g_loss.item()
        