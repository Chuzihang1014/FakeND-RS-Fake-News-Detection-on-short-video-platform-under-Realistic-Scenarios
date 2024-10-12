import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from transformers import BertModel, BertConfig
from .coattention import *
from .layers import *
from transformers import BertModel, BertConfig
import copy
import random
import numpy as np

class InvarianceEncoder(nn.Module):
    def __init__(self):
        super(InvarianceEncoder, self).__init__()

        self.visual = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))

        self.audio = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))

        self.text = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.1))

        self.co_attention = co_attention(d_k=128, d_v=128, n_heads=4, dropout=0.1,
                                            d_model=128,
                                            visual_len=83, sen_len=512, fea_v=128,
                                            fea_s=128, pos=False)



    def forward(self, fea_text, fea_comment, fea_intro, fea_img, fea_video, fea_audio, modality_type):


        if modality_type == 'text':
            fea_visual = torch.cat((fea_img, fea_video), dim=1)
            fea_visual, fea_audio = self.co_attention(v=fea_visual, s=fea_audio, v_len=fea_visual.shape[1],
                                                         s_len=fea_audio.shape[1])
            fea_visual = self.visual(fea_visual)
            fea_audio = self.audio(fea_audio)
            return fea_visual, fea_audio

        elif modality_type == 'video':

            fea_text = torch.cat((fea_text, fea_comment, fea_intro), dim=1)
            fea_text, fea_audio = self.co_attention(v=fea_text, s=fea_audio, v_len=fea_text.shape[1],
                                                         s_len=fea_audio.shape[1])
            fea_text = self.text(fea_text)
            fea_audio = self.audio(fea_audio)
            return fea_text, fea_audio

        elif modality_type == 'audio':

            fea_visual = torch.cat((fea_img, fea_video), dim=1)
            fea_visual, fea_text = self.co_attention(v=fea_visual, s=fea_text, v_len=fea_visual.shape[1],
                                                      s_len=fea_text.shape[1])

            fea_text = self.text(fea_text)
            fea_visual = self.visual(fea_visual)
            return fea_text, fea_visual

        elif modality_type == 'ta':
            fea_visual = torch.cat((fea_img, fea_video), dim=1)
            return self.visual(fea_visual), 0
        elif modality_type == 'tv':
            return self.audio(fea_audio), 0
        elif modality_type == 'va':
            fea_text = torch.cat((fea_text, fea_comment, fea_intro), dim=1)
            return self.text(fea_text), 0

class Pretrained_project(nn.Module):
    def __init__(self):
        super(Pretrained_project, self).__init__()
        self.linear_text = nn.Sequential(torch.nn.Linear(768, 128), torch.nn.LeakyReLU(),
                                         nn.Dropout(p=0.1))
        self.linear_comment = nn.Sequential(torch.nn.Linear(768, 128), torch.nn.LeakyReLU(),
                                            nn.Dropout(p=0.1))
        self.linear_intro = nn.Sequential(torch.nn.Linear(768, 128), torch.nn.LeakyReLU(),
                                          nn.Dropout(p=0.1))
        self.linear_img = nn.Sequential(torch.nn.Linear(4096, 128), torch.nn.LeakyReLU(),
                                        nn.Dropout(p=0.1))
        self.linear_video = nn.Sequential(torch.nn.Linear(4096, 128), torch.nn.LeakyReLU(),
                                          nn.Dropout(p=0.1))
        self.vggish_modified = torch.hub.load('/path/to/file', 'vggish',
                                              source='local')
        net_structure = list(self.vggish_modified.children())
        self.vggish_modified = nn.Sequential(*net_structure[-2:-1])
        self.linear_audio = nn.Sequential(torch.nn.Linear(128, 128), torch.nn.LeakyReLU(),
                                          nn.Dropout(p=0.1))

        # self.netshared = nn.Sequential()
        # self.netshared.add_module('shared_1', nn.Linear(in_features=128, out_features=128))
        # self.netshared.add_module('shared_1_activation', nn.LeakyReLU())
        # self.netshared.add_module('shared_1_dropout', nn.Dropout(p=0.1))
        # self.netshared.add_module('shared_2', nn.Linear(in_features=128, out_features=128))
        # self.netshared.add_module('shared_2_activation', nn.LeakyReLU())
        # self.netshared.add_module('shared_2_dropout', nn.Dropout(p=0.1))
        # self.netshared.add_module('shared_3', nn.Linear(in_features=128, out_features=128))
        # self.netshared.add_module('shared_3_activation', nn.LeakyReLU())
        # self.netshared.add_module('shared_3_dropout', nn.Dropout(p=0.1))
        # self.netshared.add_module('shared_4', nn.Linear(in_features=128, out_features=128))
        # self.netshared.add_module('shared_4_activation', nn.LeakyReLU())
        # self.netshared.add_module('shared_4_dropout', nn.Dropout(p=0.1))


    def forward(self, text, comment, intro, frames, c3d, audioframes):

        text = self.linear_text(text)
        comment = self.linear_comment(comment)
        intro = self.linear_intro(intro)
        frames = self.linear_img(frames)
        c3d = self.linear_video(c3d)
        audioframes = self.vggish_modified(audioframes)
        audioframes = self.linear_audio(audioframes)


        return torch.mean(text, -2) + comment + intro, torch.mean(frames, -2) + torch.mean(c3d, -2), torch.mean(audioframes, -2)

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


