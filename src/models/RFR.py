import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import copy
import random
import numpy as np



class Autoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(p=0.1)
        self.activation = nn.LeakyReLU()


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.dropout(self.activation(self.decoder(encoded)))

        return decoded

class IF_IM(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(IF_IM, self).__init__()
        self.autoencoders = nn.ModuleList([Autoencoder(input_size, hidden_sizes[i]) for i in range(len(hidden_sizes))])
        self.prompt = nn.Parameter(torch.randn(1, 128), requires_grad=True).to('cuda')

        self.linear = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1))

    def forward(self, h):

        delta_z_list = []
        delta_z = self.autoencoders[0](torch.cat((h, self.prompt.repeat(h.shape[0], 1)), dim=1))
        delta_z_list.append(delta_z)

        for i in range(1, len(self.autoencoders)):
            delta_z = self.autoencoders[i](delta_z_list[i-1])
            delta_z_list.append(delta_z)

        h_imagined = self.linear(delta_z_list[-1])


        return h_imagined

class CMD(nn.Module):

    def __init__(self):
        super(CMD, self).__init__()

    def forward(self, x1, x2, n_moments):
        mx1 = torch.mean(x1, 0)
        mx2 = torch.mean(x2, 0)
        sx1 = x1-mx1
        sx2 = x2-mx2
        dm = self.matchnorm(mx1, mx2)
        scms = dm
        for i in range(n_moments - 1):
            scms += self.scm(sx1, sx2, i + 2)
        return scms

    def matchnorm(self, x1, x2):
        power = torch.pow(x1-x2,2)
        summed = torch.sum(power)
        sqrt = summed**(0.5)
        return sqrt
        # return ((x1-x2)**2).sum().sqrt()

    def scm(self, sx1, sx2, k):
        ss1 = torch.mean(torch.pow(sx1, k), 0)
        ss2 = torch.mean(torch.pow(sx2, k), 0)
        return self.matchnorm(ss1, ss2)

class LossFunctions(nn.Module):
    def __init__(self,):
        super(LossFunctions, self).__init__()  # 调用父类的构造函数
        self.loss_cmd_func = CMD()

    def shared_loss(self, invariance_x, invariance_y):
        # losses between shared states
        loss = self.loss_cmd_func(invariance_x, invariance_y, 2)

        return loss


    def spec_loss(self, H_predicted, H_target):

        return F.mse_loss(H_predicted, H_target)



    def distill_loss(self, text, visual, audio, reconstructed_x, reconstructed_y, modality_type):

        sim_tv = F.cosine_similarity(text, visual, dim=-1)
        sim_ta = F.cosine_similarity(text, audio, dim=-1)
        sim_va = F.cosine_similarity(visual, audio, dim=-1)
        if modality_type == 'text':
            sim_xv = F.cosine_similarity(visual, reconstructed_x, dim=-1)
            sim_xa = F.cosine_similarity(audio, reconstructed_x, dim=-1)
            loss = F.mse_loss(sim_tv, sim_xv) + F.mse_loss(sim_ta, sim_xa)

        elif modality_type == 'visual':
            sim_xt = F.cosine_similarity(text, visual, dim=-1)
            sim_xa = F.cosine_similarity(reconstructed_x, audio, dim=-1)
            loss = F.mse_loss(sim_tv, sim_xt) + F.mse_loss(sim_va, sim_xa)

        elif modality_type == 'audio':
            sim_xt = F.cosine_similarity(text, reconstructed_x, dim=-1)
            sim_xv = F.cosine_similarity(visual, reconstructed_x, dim=-1)
            loss = F.mse_loss(sim_va, sim_xv) + F.mse_loss(sim_ta, sim_xt)

        elif modality_type == 'ta':
            sim_xv = F.cosine_similarity(visual, reconstructed_x, dim=-1)
            sim_yv = F.cosine_similarity(visual, reconstructed_y, dim=-1)
            loss = F.mse_loss(sim_va, sim_xv) + F.mse_loss(sim_tv, sim_yv)

        elif modality_type == 'tv':
            sim_xa = F.cosine_similarity(audio, reconstructed_x, dim=-1)
            sim_ya = F.cosine_similarity(audio, reconstructed_y, dim=-1)
            loss = F.mse_loss(sim_va, sim_xa) + F.mse_loss(sim_ta, sim_ya)

        elif modality_type == 'va':
            sim_xt = F.cosine_similarity(text, reconstructed_x, dim=-1)
            sim_yt = F.cosine_similarity(text, reconstructed_y, dim=-1)
            loss = F.mse_loss(sim_ta, sim_xt) + F.mse_loss(sim_tv, sim_yt)

        return loss




