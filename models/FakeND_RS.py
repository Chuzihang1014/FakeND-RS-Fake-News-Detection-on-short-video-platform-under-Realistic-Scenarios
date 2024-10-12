import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import *
from tqdm import tqdm
from transformers import AutoConfig, BertModel
from transformers.models.bert.modeling_bert import BertLayer
from zmq import device

from .coattention import *
from .layers import *
from utils.metrics import *
from transformers import BertModel, BertConfig
from transformers import BertTokenizer
from .scrip import *
from .RFR import *


class FakeND_RSModel(torch.nn.Module):
    def __init__(self, bert_model, fea_dim, dropout):
        super(FakeND_RSModel, self).__init__()

        # 替换为你的本地路径
        model_path = "/path/to/file"

        # 加载配置文件
        config = BertConfig.from_json_file(model_path + "config.json")

        # 直接加载本地预训练的 BERT 模型，并设置为不可训练
        self.bert = BertModel.from_pretrained(model_path, config=config).requires_grad_(False)


        self.text_dim = 768
        self.comment_dim = 768
        self.img_dim = 4096
        self.video_dim = 4096
        self.num_frames = 83
        self.num_audioframes = 50
        self.num_comments = 23
        self.dim = fea_dim
        self.num_heads = 4

        self.dropout = dropout

        self.attention = Attention(dim=self.dim, heads=4, dropout=dropout)

        self.vggish_layer = torch.hub.load('/path/to/file', 'vggish',
                                           source='local')
        net_structure = list(self.vggish_layer.children())
        self.vggish_modified = nn.Sequential(*net_structure[-2:-1])

        self.co_attention_ta = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout,
                                            d_model=fea_dim,
                                            visual_len=1, sen_len=512, fea_v=self.dim,
                                            fea_s=self.dim, pos=False)
        self.co_attention_tv = co_attention(d_k=fea_dim, d_v=fea_dim, n_heads=self.num_heads, dropout=self.dropout,
                                            d_model=fea_dim,
                                            visual_len=self.num_frames, sen_len=512, fea_v=self.dim, fea_s=self.dim,
                                            pos=False)

        self.trm = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.dim, nhead=2), num_layers=1)  # 改过

        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim), torch.nn.LeakyReLU(),
                                         nn.Dropout(p=self.dropout))
        self.linear_comment = nn.Sequential(torch.nn.Linear(self.comment_dim, fea_dim), torch.nn.LeakyReLU(),
                                            nn.Dropout(p=self.dropout))
        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, fea_dim), torch.nn.LeakyReLU(),
                                        nn.Dropout(p=self.dropout))
        self.linear_video = nn.Sequential(torch.nn.Linear(self.video_dim, fea_dim), torch.nn.LeakyReLU(),
                                          nn.Dropout(p=self.dropout))
        self.linear_intro = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim), torch.nn.LeakyReLU(),
                                          nn.Dropout(p=self.dropout))
        self.linear_audio = nn.Sequential(torch.nn.Linear(128, fea_dim), torch.nn.LeakyReLU(),
                                          nn.Dropout(p=self.dropout))


        self.model_invariance = InvarianceEncoder()
        self.pretrained_project = Pretrained_project()
        checkpoint_path = "/path/to/file"  # 模型参数保存的路径
        checkpoint = torch.load(checkpoint_path)
        self.pretrained_project.load_state_dict(checkpoint)
        # 将整个状态字典加载到模型中
        for param in self.pretrained_project.parameters():
            param.requires_grad = False
        self.loss = LossFunctions()
        self.IF_IM_text = IF_IM(256, [128, 256, 128])
        self.IF_IM_visual = IF_IM(256, [128, 256, 128])
        self.IF_IM_audio = IF_IM(256, [128, 256, 128])


        self.classifier = nn.Linear(fea_dim, 2)



    def forward(self, train_or_test, modality_type, **kwargs):


        ### User Intro ###
        intro_inputid = kwargs['intro_inputid']
        intro_mask = kwargs['intro_mask']
        intro = self.bert(intro_inputid, attention_mask=intro_mask)[1]
        fea_intro = self.linear_intro(intro)


        ### Title ###
        title_inputid = kwargs['title_inputid']  # (batch,512)
        title_mask = kwargs['title_mask']  # (batch,512)
        text = self.bert(title_inputid, attention_mask=title_mask)['last_hidden_state']  # (batch,sequence,768)
        fea_text = self.linear_text(text)

        ### Audio Frames ###
        audioframes = kwargs['audioframes']  # (batch,36,12288)
        audioframes_masks = kwargs['audioframes_masks']
        fea_audio = self.vggish_modified(audioframes)  # (batch, frames, 128)
        fea_audio = self.linear_audio(fea_audio)

        ### Image Frames ###
        frames = kwargs['frames']  # (batch,30,4096)
        frames_masks = kwargs['frames_masks']
        fea_img = self.linear_img(frames)

        ### C3D ###
        c3d = kwargs['c3d']  # (batch, 36, 4096)
        c3d_masks = kwargs['c3d_masks']
        fea_video = self.linear_video(c3d)  # (batch, frames, 128)

        ### Comment ###
        comments_inputid = kwargs['comments_inputid']  # (batch,20,250)
        comments_mask = kwargs['comments_mask']  # (batch,20,250)

        comments_like = kwargs['comments_like']
        comments_feature = []
        for i in range(comments_inputid.shape[0]):
            bert_fea = self.bert(comments_inputid[i], attention_mask=comments_mask[i])[1]
            comments_feature.append(bert_fea)
        comments_feature = torch.stack(comments_feature)  # (batch,seq,fea_dim)

        fea_comments = []
        for v in range(comments_like.shape[0]):
            comments_weight = torch.stack(
                [torch.true_divide((i + 1), (comments_like[v].shape[0] + comments_like[v].sum())) for i in
                 comments_like[v]])
            comments_fea_reweight = torch.sum(
                comments_feature[v] * (comments_weight.reshape(comments_weight.shape[0], 1)), dim=0)
            fea_comments.append(comments_fea_reweight)
        fea_comment = torch.stack(fea_comments)
        fea_comments = self.linear_comment(fea_comment)  # (batch,fea_dim)


        if train_or_test == 'train':

            if modality_type == 'text':

                invariance_x, invariance_y = self.model_invariance(
                    fea_text,
                    fea_comments.unsqueeze(1),
                    fea_intro.unsqueeze(1),
                    fea_img,
                    fea_video,
                    fea_audio,
                    modality_type)

                fea_invariance = torch.cat((invariance_x, invariance_y),dim=1)
                fea_invariance = torch.mean(fea_invariance, dim=1)


                h_imaged_text = self.IF_IM_text(fea_invariance)

                text, visual, audio = self.pretrained_project(text,
                                                              fea_comment,
                                                              intro,
                                                              frames,
                                                              c3d,
                                                              audioframes)


                loss_sha = self.loss.shared_loss(torch.mean(invariance_x, dim=1), torch.mean(invariance_y, dim=1))
                loss_spec = self.loss.spec_loss(h_imaged_text, torch.mean(fea_text, dim=1)+fea_comments+fea_intro)
                loss_distill = self.loss.distill_loss(text, visual, audio, h_imaged_text, None, modality_type)

                fea_text = h_imaged_text
                fea_visual = torch.mean(fea_img, dim=-2) + torch.mean(fea_video, dim=-2)
                fea_audio = torch.mean(fea_audio, dim=-2)

            if modality_type == 'video':

                invariance_x, invariance_y = self.model_invariance(
                    fea_text,
                    fea_comments.unsqueeze(1),
                    fea_intro.unsqueeze(1),
                    fea_img,
                    fea_video,
                    fea_audio,
                    modality_type)

                fea_invariance = torch.cat((invariance_x, invariance_y), dim=1)
                fea_invariance = torch.mean(fea_invariance, dim=1)

                h_imaged_visual = self.IF_IM_visual(fea_invariance)

                text, visual, audio = self.pretrained_project(text,
                                                              fea_comment,
                                                              intro,
                                                              frames,
                                                              c3d,
                                                              audioframes)

                loss_sha = self.loss.shared_loss(torch.mean(invariance_x, dim=1), torch.mean(invariance_y, dim=1))
                loss_spec = self.loss.spec_loss(h_imaged_visual, torch.mean(fea_img, dim=1) + torch.mean(fea_video, dim=1))
                loss_distill = self.loss.distill_loss(text, visual, audio, h_imaged_visual, None, modality_type)

                fea_text = torch.mean(fea_text, dim=-2) + fea_comments + fea_intro
                fea_visual = h_imaged_visual
                fea_audio = torch.mean(fea_audio, dim=-2)

            if modality_type == 'audio':
                invariance_x, invariance_y = self.model_invariance(
                    fea_text,
                    fea_comments.unsqueeze(1),
                    fea_intro.unsqueeze(1),
                    fea_img,
                    fea_video,
                    fea_audio,
                    modality_type)

                fea_invariance = torch.cat((invariance_x, invariance_y), dim=1)
                fea_invariance = torch.mean(fea_invariance, dim=1)

                h_imaged_audio = self.IF_IM_audio(fea_invariance)

                text, visual, audio = self.pretrained_project(text,
                                                              fea_comment,
                                                              intro,
                                                              frames,
                                                              c3d,
                                                              audioframes)

                loss_sha = self.loss.shared_loss(torch.mean(invariance_x, dim=1), torch.mean(invariance_y, dim=1))
                loss_spec = self.loss.spec_loss(h_imaged_audio, torch.mean(fea_audio, dim=1))
                loss_distill = self.loss.distill_loss(text, visual, audio, h_imaged_audio, None, modality_type)

                fea_text = torch.mean(fea_text, dim=-2) + fea_comments + fea_intro
                fea_visual = torch.mean(fea_img, dim=1) + torch.mean(fea_video, dim=1)
                fea_audio = h_imaged_audio

            if modality_type == 'tv':

                fea_invariance = torch.mean(fea_audio, dim=1)

                h_imaged_text = self.IF_IM_text(fea_invariance)
                h_imaged_visual = self.IF_IM_visual(fea_invariance)

                text, visual, audio = self.pretrained_project(text,
                                                              fea_comment,
                                                              intro,
                                                              frames,
                                                              c3d,
                                                              audioframes)

                loss_sha = 0
                loss_spec_text = self.loss.spec_loss(h_imaged_text, torch.mean(fea_text, dim=1) + fea_comments + fea_intro)
                loss_spec_visual = self.loss.spec_loss(h_imaged_visual, torch.mean(fea_img, dim=1) + torch.mean(fea_video, dim=1))
                loss_spec = loss_spec_text + loss_spec_visual

                loss_distill = self.loss.distill_loss(text, visual, audio, h_imaged_visual, h_imaged_text, modality_type)

                fea_text = h_imaged_text
                fea_visual = h_imaged_visual
                fea_audio = torch.mean(fea_audio, dim=1)

            if modality_type == 'va':
                fea_invariance = torch.mean(fea_text, dim=1) + fea_comments + fea_intro

                h_imaged_audio = self.IF_IM_audio(fea_invariance)
                h_imaged_visual = self.IF_IM_visual(fea_invariance)

                text, visual, audio = self.pretrained_project(text,
                                                              fea_comment,
                                                              intro,
                                                              frames,
                                                              c3d,
                                                              audioframes)

                loss_sha = 0
                loss_spec_visual = self.loss.spec_loss(h_imaged_visual,
                                                     torch.mean(fea_img, dim=1) + torch.mean(fea_video, dim=1))
                loss_spec_audio = self.loss.spec_loss(h_imaged_audio, torch.mean(fea_audio, dim=1))
                loss_spec = loss_spec_audio + loss_spec_visual

                loss_distill = self.loss.distill_loss(text, visual, audio, h_imaged_audio, h_imaged_visual, modality_type)

                fea_text = torch.mean(fea_text, dim=1) + fea_comments + fea_intro
                fea_visual = h_imaged_visual
                fea_audio = h_imaged_audio

            if modality_type == 'ta':
                fea_invariance = torch.mean(fea_img, dim=1) + torch.mean(fea_video, dim=1)

                h_imaged_audio = self.IF_IM_audio(fea_invariance)
                h_imaged_text = self.IF_IM_text(fea_invariance)

                text, visual, audio = self.pretrained_project(text,
                                                              fea_comment,
                                                              intro,
                                                              frames,
                                                              c3d,
                                                              audioframes)

                loss_sha = 0
                loss_spec_text = self.loss.spec_loss(h_imaged_text,
                                                       torch.mean(fea_text, dim=1) + fea_comments + fea_intro)
                loss_spec_audio = self.loss.spec_loss(h_imaged_audio, torch.mean(fea_audio, dim=1))
                loss_spec = loss_spec_audio + loss_spec_text

                loss_distill = self.loss.distill_loss(text, visual, audio, h_imaged_audio, h_imaged_text, modality_type)

                fea_text = h_imaged_text
                fea_visual = torch.mean(fea_img ,dim=1) + torch.mean(fea_video, dim=1)
                fea_audio = h_imaged_audio

        else :

             if modality_type == 'text':
                 invariance_x, invariance_y = self.model_invariance(
                     fea_text,
                     fea_comments.unsqueeze(1),
                     fea_intro.unsqueeze(1),
                     fea_img,
                     fea_video,
                     fea_audio,
                     modality_type)

                 fea_invariance = torch.cat((invariance_x, invariance_y), dim=1)
                 fea_invariance = torch.mean(fea_invariance, dim=1)

                 h_imaged_text = self.IF_IM_text(fea_invariance)

                 text, visual, audio = self.pretrained_project(text,
                                                               fea_comment,
                                                               intro,
                                                               frames,
                                                               c3d,
                                                               audioframes)

                 loss_sha = self.loss.shared_loss(torch.mean(invariance_x, dim=1), torch.mean(invariance_y, dim=1))
                 loss_spec = self.loss.spec_loss(h_imaged_text, torch.mean(fea_text, dim=1) + fea_comments + fea_intro)
                 loss_distill = self.loss.distill_loss(text, visual, audio, h_imaged_text, None, modality_type)

                 fea_text = h_imaged_text
                 fea_visual = torch.mean(fea_img, dim=-2) + torch.mean(fea_video, dim=-2)
                 fea_audio = torch.mean(fea_audio, dim=-2)

             if modality_type == 'video':
                 invariance_x, invariance_y = self.model_invariance(
                     fea_text,
                     fea_comments.unsqueeze(1),
                     fea_intro.unsqueeze(1),
                     fea_img,
                     fea_video,
                     fea_audio,
                     modality_type)

                 fea_invariance = torch.cat((invariance_x, invariance_y), dim=1)
                 fea_invariance = torch.mean(fea_invariance, dim=1)

                 h_imaged_visual = self.IF_IM_visual(fea_invariance)

                 text, visual, audio = self.pretrained_project(text,
                                                               fea_comment,
                                                               intro,
                                                               frames,
                                                               c3d,
                                                               audioframes)

                 loss_sha = self.loss.shared_loss(torch.mean(invariance_x, dim=1), torch.mean(invariance_y, dim=1))
                 loss_spec = self.loss.spec_loss(h_imaged_visual,
                                                 torch.mean(fea_img, dim=1) + torch.mean(fea_video, dim=1))
                 loss_distill = self.loss.distill_loss(text, visual, audio, h_imaged_visual, None, modality_type)

                 fea_text = torch.mean(fea_text, dim=-2) + fea_comments + fea_intro
                 fea_visual = h_imaged_visual
                 fea_audio = torch.mean(fea_audio, dim=-2)

             if modality_type == 'audio':
                 invariance_x, invariance_y = self.model_invariance(
                     fea_text,
                     fea_comments.unsqueeze(1),
                     fea_intro.unsqueeze(1),
                     fea_img,
                     fea_video,
                     fea_audio,
                     modality_type)

                 fea_invariance = torch.cat((invariance_x, invariance_y), dim=1)
                 fea_invariance = torch.mean(fea_invariance, dim=1)

                 h_imaged_audio = self.IF_IM_audio(fea_invariance)

                 text, visual, audio = self.pretrained_project(text,
                                                               fea_comment,
                                                               intro,
                                                               frames,
                                                               c3d,
                                                               audioframes)

                 loss_sha = self.loss.shared_loss(torch.mean(invariance_x, dim=1), torch.mean(invariance_y, dim=1))
                 loss_spec = self.loss.spec_loss(h_imaged_audio, torch.mean(fea_audio, dim=1))
                 loss_distill = self.loss.distill_loss(text, visual, audio, h_imaged_audio, None, modality_type)

                 fea_text = torch.mean(fea_text, dim=-2) + fea_comments + fea_intro
                 fea_visual = torch.mean(fea_img, dim=1) + torch.mean(fea_video, dim=1)
                 fea_audio = h_imaged_audio

             if modality_type == 'tv':
                 fea_invariance = torch.mean(fea_audio, dim=1)

                 h_imaged_text = self.IF_IM_text(fea_invariance)
                 h_imaged_visual = self.IF_IM_visual(fea_invariance)

                 text, visual, audio = self.pretrained_project(text,
                                                               fea_comment,
                                                               intro,
                                                               frames,
                                                               c3d,
                                                               audioframes)

                 loss_sha = 0
                 loss_spec_text = self.loss.spec_loss(h_imaged_text,
                                                      torch.mean(fea_text, dim=1) + fea_comments + fea_intro)
                 loss_spec_visual = self.loss.spec_loss(h_imaged_visual,
                                                        torch.mean(fea_img, dim=1) + torch.mean(fea_video, dim=1))
                 loss_spec = loss_spec_text + loss_spec_visual

                 loss_distill = self.loss.distill_loss(text, visual, audio, h_imaged_visual, h_imaged_text, modality_type)

                 fea_text = h_imaged_text
                 fea_visual = h_imaged_visual
                 fea_audio = torch.mean(fea_audio, dim=1)

             if modality_type == 'va':
                 fea_invariance = torch.mean(fea_text, dim=1) + fea_comments + fea_intro

                 h_imaged_audio = self.IF_IM_audio(fea_invariance)
                 h_imaged_visual = self.IF_IM_visual(fea_invariance)

                 text, visual, audio = self.pretrained_project(text,
                                                               fea_comment,
                                                               intro,
                                                               frames,
                                                               c3d,
                                                               audioframes)

                 loss_sha = 0
                 loss_spec_visual = self.loss.spec_loss(h_imaged_visual,
                                                        torch.mean(fea_img, dim=1) + torch.mean(fea_video, dim=1))
                 loss_spec_audio = self.loss.spec_loss(h_imaged_audio, torch.mean(fea_audio, dim=1))
                 loss_spec = loss_spec_audio + loss_spec_visual

                 loss_distill = self.loss.distill_loss(text, visual, audio, h_imaged_audio, h_imaged_visual, modality_type)

                 fea_text = torch.mean(fea_text, dim=1) + fea_comments + fea_intro
                 fea_visual = h_imaged_visual
                 fea_audio = h_imaged_audio

             if modality_type == 'ta':
                 fea_invariance = torch.mean(fea_img, dim=1) + torch.mean(fea_video, dim=1)

                 h_imaged_audio = self.IF_IM_audio(fea_invariance)
                 h_imaged_text = self.IF_IM_text(fea_invariance)

                 text, visual, audio = self.pretrained_project(text,
                                                               fea_comment,
                                                               intro,
                                                               frames,
                                                               c3d,
                                                               audioframes)

                 loss_sha = 0
                 loss_spec_text = self.loss.spec_loss(h_imaged_text,
                                                      torch.mean(fea_text, dim=1) + fea_comments + fea_intro)
                 loss_spec_audio = self.loss.spec_loss(h_imaged_audio, torch.mean(fea_audio, dim=1))
                 loss_spec = loss_spec_audio + loss_spec_text

                 loss_distill = self.loss.distill_loss(text, visual, audio, h_imaged_audio, h_imaged_text, modality_type)

                 fea_text = h_imaged_text
                 fea_visual = torch.mean(fea_img, dim=1) + torch.mean(fea_video, dim=1)
                 fea_audio = h_imaged_audio


        output_text = self.classifier(fea_text)
        output_visual = self.classifier(fea_visual)
        output_audio = self.classifier(fea_audio)
        output = output_text + output_visual + output_audio

        return output, output_text, output_visual, output_audio, loss_sha, loss_spec, loss_distill