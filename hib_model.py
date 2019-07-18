import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_util as ptu
from torch.nn.parameter import Parameter

class HIB_model(torch.nn.Module):

    def __init__(self,input_size=5,z_dimension=2,beta=1e-3,lr=1e-3,device=3):
        super(HIB_model,self).__init__()
        self.device = device
        self.input_size = input_size
        self.z_dimension = z_dimension
        self.beta = beta
        self.lr = lr
        self.enc = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
        )
        self.enc_mu = nn.Linear(32, z_dimension)
        self.enc_var_square = nn.Sequential(nn.Linear(32, z_dimension),nn.Softplus())

        self.scalar_params = nn.Linear(1, 1)

        params = (list(self.enc.parameters()) +
                  list(self.enc_mu.parameters()) +
                  list(self.enc_var_square.parameters())+list(self.scalar_params.parameters()))

        self.cuda(self.device)
        self.optimizer = optim.Adam(params, lr=self.lr)
        self.num_z_sample = 8

    def get_output(self,input):
        enc_data = self.enc(input)
        mu = self.enc_mu(enc_data)
        var = self.enc_var_square(enc_data)
        return mu,var

    def _product_of_gaussians(self,mu,var):
        var = torch.clamp(var, min=1e-7)
        var = 1. / torch.sum(torch.reciprocal(var), dim=0)
        mu = var * torch.sum(mu / var, dim=0)
        return mu, var



    def cal_z(self,data):
        z_mean = torch.zeros((data.shape[0],self.z_dimension)).cuda(self.device)
        z_var_square = torch.zeros((data.shape[0],self.z_dimension)).cuda(self.device)
        for i in range(data.shape[0]):
            input = data[i,:,:].permute(1,0)
            mu,var = self.get_output(input)
            mu_multi, var_multi = self._product_of_gaussians(mu,var)
            z_mean[i,:] = mu_multi
            z_var_square[i,:] = var_multi
        return z_mean, z_var_square


    def kl_loss(self,mean,var):
        prior = torch.distributions.Normal(torch.zeros(self.z_dimension).cuda(self.device), torch.ones(self.z_dimension).cuda(self.device))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in
                      zip(torch.unbind(mean), torch.unbind(var))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_mean = torch.mean(torch.stack(kl_divs))
        return kl_div_mean

    def soft_z_loss(self,z_mean_1, z_var_1,z_mean_2, z_var_2,labels):
        loss_1 = torch.zeros((z_mean_1.shape[0],1)).cuda(self.device)
        loss_0 = torch.zeros((z_mean_1.shape[0], 1)).cuda(self.device)
        for i in range(z_mean_1.shape[0]):
            z1_dis = torch.distributions.Normal(z_mean_1[i,:],torch.sqrt(z_var_1[i,:],))
            z1_sample = z1_dis.rsample((self.num_z_sample,)).cuda(self.device)
            z2_dis = torch.distributions.Normal(z_mean_2[i, :], torch.sqrt(z_var_2[i, :], ))
            z2_sample = z2_dis.rsample((self.num_z_sample,)).cuda(self.device)
            loss1=torch.zeros((self.num_z_sample,self.num_z_sample)).cuda(self.device)
            loss0=torch.zeros((self.num_z_sample,self.num_z_sample)).cuda(self.device)
            for j in range(self.num_z_sample):
                for k in range(self.num_z_sample):
                    dis = torch.norm(z1_sample[j,:]-z2_sample[k,:]).cuda(self.device)
                    possibility = torch.sigmoid(-1*F.softplus(self.scalar_params.weight)*dis+self.scalar_params.bias).cuda(self.device)
                    loss1[j,k]=-1*torch.log(possibility)
                    loss0[j,k]-1*torch.log(1-possibility)
            loss_1[i,0] = torch.mean(loss1)
            loss_0[i, 0] = torch.mean(loss0)
        loss = torch.mean(loss_1 * labels + loss_0 * (1-labels))
        return loss


    def cal_loss(self,batch,labels):
        z_mean_1, z_var_1 = self.cal_z(batch[:,:,:,0])
        z_mean_2, z_var_2 = self.cal_z(batch[:, :, :, 1])
        loss = (self.kl_loss(z_mean_1,z_var_1) + self.kl_loss(z_mean_2, z_var_2))*self.beta
        loss = loss + self.soft_z_loss(z_mean_1, z_var_1,z_mean_2, z_var_2,labels)
        return loss


    def _confidence(self,mean,var):
        poss_total = torch.zeros((mean.shape[0],1)).cuda(self.device)
        for i in range(mean.shape[0]):
            z_dis = torch.distributions.Normal(mean[i, :], torch.sqrt(var[i, :], ))
            z1_sample = z_dis.rsample((self.num_z_sample,)).cuda(self.device)
            z2_sample = z_dis.rsample((self.num_z_sample,)).cuda(self.device)
            poss = torch.zeros((self.num_z_sample, self.num_z_sample)).cuda(self.device)
            for j in range(self.num_z_sample):
                for k in range(self.num_z_sample):
                    dis = torch.norm(z1_sample[j, :] - z2_sample[k, :]).cuda(self.device)
                    possibility = torch.sigmoid(
                        -1 * F.softplus(self.scalar_params.weight) * dis + self.scalar_params.bias).cuda(self.device)
                    poss[j, k] = possibility

            poss_total[i, 0] = torch.mean(poss)

        return poss_total

    def cal_confidence(self,batch):#(batch_size,obs_dim,tra_len)
        z_mean, z_var = self.cal_z(batch)
        confidence = self._confidence(z_mean,z_var)
        return confidence

    def optimize(self,loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


