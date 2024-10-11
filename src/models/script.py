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

class AGC(nn.Module):
    def __init__(self, optimizer, reduction='sum'):
        self._optim, self._reduction = optimizer, reduction
        self.iter = 0

    @property
    def optimizer(self,):
        return self._optim

    def zero_grad(self,):


        return self._optim.zero_grad(set_to_none=True)

    def pc_backward(self, t, v, a, label, objectives, ddp_model=None):

        grads, shapes, has_grads = self._pack_grad(objectives, ddp_model)
        Gall = self.adaptive_gradient_calibration(grads, t, v, a, label)
        pc_grad = self._unflatten_grad(Gall, shapes[0])
        # self._set_grad(pc_grad)
        return pc_grad

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        coefs = torch.ones(num_task, dtype=torch.float32, device=grads[0].device)
        for g_i in pc_grad:
            indices = list(range(num_task))
            random.shuffle(list(range(num_task)))
            random.shuffle(grads)
            for index in indices:
                g_j = grads[index]
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    coef = g_i_g_j / (g_j.norm() ** 2)

                    g_i -= coef * g_j
                    coefs[index] -= coef
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)

        self.iter += 1
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives, ddp):

        grads, shapes, has_grads = [], [], []
        for ii, obj in enumerate(objectives):
            self._optim.zero_grad(set_to_none=True)
            # if ii == 0: continue
            # out_tensors = list(_find_tensors(obj))
            # ddp.reducer.prepare_for_backward(out_tensors)
            if ii < len(objectives) - 1:
                obj.backward(retain_graph=True)
            else:
                obj.backward(retain_graph=False)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self,):

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

    def adaptive_gradient_calibration(self, grad, t, v, a, y_true):


        if y_true == 0:

           first_values = torch.tensor([t[0, 0], v[0, 0], a[0, 0]])

           # 找到最大值的索引
           max_index = torch.argmax(first_values)

           # 根据索引选择相应的张量
           if max_index == 0:
               guidance = grad[0].unsqueeze(0)
               cos_tv = F.cosine_similarity(grad[1].unsqueeze(0), guidance)
               cos_av = F.cosine_similarity(grad[2].unsqueeze(0), guidance)
               G_t_projected = grad[1] * cos_tv if cos_tv >= 0 else torch.zeros_like(grad[1])
               G_a_projected = grad[2] * cos_av if cos_av >= 0 else torch.zeros_like(grad[2])
               Gall = guidance + G_t_projected + G_a_projected
           elif max_index == 1:
               guidance = grad[1].unsqueeze(0)
               cos_tv = F.cosine_similarity(grad[0].unsqueeze(0), guidance)
               cos_ta = F.cosine_similarity(grad[2].unsqueeze(0), guidance)
               G_v_projected = grad[0] * cos_tv if cos_tv >= 0 else torch.zeros_like(grad[0])
               G_a_projected = grad[2] * cos_ta if cos_ta >= 0 else torch.zeros_like(grad[2])
               Gall = guidance + G_v_projected + G_a_projected
           else:
               guidance = grad[2].unsqueeze(0)
               cos_av = F.cosine_similarity(grad[0].unsqueeze(0), guidance)
               cos_ta = F.cosine_similarity(grad[1].unsqueeze(0), guidance)
               G_v_projected = grad[0] * cos_av if cos_av >= 0 else torch.zeros_like(grad[0])
               G_t_projected = grad[1] * cos_ta if cos_ta >= 0 else torch.zeros_like(grad[1])
               Gall = guidance + G_v_projected + G_t_projected

        if y_true == 1:
            first_values = torch.tensor([t[0, 1], v[0, 1], a[0, 1]])

            # 找到最大值的索引
            max_index = torch.argmax(first_values)

            if max_index == 0:
                guidance = grad[0].unsqueeze(0)
                cos_tv = F.cosine_similarity(grad[1].unsqueeze(0), guidance)
                cos_av = F.cosine_similarity(grad[2].unsqueeze(0), guidance)
                G_t_projected = grad[1] * cos_tv if cos_tv >= 0 else torch.zeros_like(grad[1])
                G_a_projected = grad[2] * cos_av if cos_av >= 0 else torch.zeros_like(grad[2])
                Gall = guidance + G_t_projected + G_a_projected
            elif max_index == 1:
                guidance = grad[1].unsqueeze(0)
                cos_tv = F.cosine_similarity(grad[0].unsqueeze(0), guidance)
                cos_ta = F.cosine_similarity(grad[2].unsqueeze(0), guidance)
                G_v_projected = grad[0] * cos_tv if cos_tv >= 0 else torch.zeros_like(grad[0])
                G_a_projected = grad[2] * cos_ta if cos_ta >= 0 else torch.zeros_like(grad[2])
                Gall = guidance + G_v_projected + G_a_projected
            else:
                guidance = grad[2].unsqueeze(0)
                cos_av = F.cosine_similarity(grad[0].unsqueeze(0), guidance)
                cos_ta = F.cosine_similarity(grad[1].unsqueeze(0), guidance)
                G_v_projected = grad[0] * cos_av if cos_av >= 0 else torch.zeros_like(grad[0])
                G_t_projected = grad[1] * cos_ta if cos_ta >= 0 else torch.zeros_like(grad[1])
                Gall = guidance + G_v_projected + G_t_projected
        Gall = list(Gall.squeeze(0))

        return Gall


