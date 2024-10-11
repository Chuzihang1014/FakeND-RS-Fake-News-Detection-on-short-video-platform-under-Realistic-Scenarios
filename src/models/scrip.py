import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from transformers import BertModel, BertConfig
from .coattention import *
from .layers import *
from transformers import BertModel, BertConfig

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




