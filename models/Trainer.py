import copy
import json
import os
import time
from tkinter import E

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tqdm
from sklearn.metrics import *
from tqdm import tqdm
from transformers import BertModel
from utils.metrics import *
from zmq import device
import wandb
from .coattention import *
from .layers import *
from .scrip import InvarianceEncoder
from .RFR import *


class Trainer():
    def __init__(self,
                model, 
                 device,
                 lr,
                 dropout,
                 dataloaders,
                 weight_decay,
                 save_param_path,
                 writer, 
                 epoch_stop,
                 epoches,
                 mode,
                 model_name, 
                 event_num,
                 modality_type,
                 save_threshold = 0.0, 
                 start_epoch = 0,
                 ):
        
        self.model = model
        self.device = device
        self.mode = mode
        self.model_name = model_name
        self.event_num = event_num

        self.dataloaders = dataloaders
        self.start_epoch = start_epoch
        self.num_epochs = epoches
        self.epoch_stop = epoch_stop
        self.save_threshold = save_threshold
        self.writer = writer
        self.modality_type = modality_type
        self.lambda_1 = 100
        self.lambda_2 = 10


        if os.path.exists(save_param_path):
            self.save_param_path = save_param_path
        else:
            self.save_param_path = os.makedirs(save_param_path)
            self.save_param_path= save_param_path

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
    
        self.criterion = nn.CrossEntropyLoss()


        

    def train(self):

        since = time.time()

        self.model.cuda()

        best_model_wts_test = copy.deepcopy(self.model.state_dict())
        best_acc_test = 0.0
        best_epoch_test = 0
        is_earlystop = False

        if self.mode == "eann":
            best_acc_test_event = 0.0
            best_epoch_test_event = 0

        for epoch in range(5, self.start_epoch+self.num_epochs):
            if is_earlystop:
                break
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch+1, self.start_epoch+self.num_epochs))
            print('-' * 50)

            p = float(epoch) / 100
            lr = self.lr / (1. + 10 * p) ** 0.75
            path = 'None'

            self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=lr)
            self.pcgrad = AGC(self.optimizer, reduction='sum')
            
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.model.train()  
                else:
                    self.model.eval()
                    if path == 'None':
                       torch.save(self.model.state_dict(), self.save_param_path + "train")
                       path = self.save_param_path + "train"
                print('-' * 10)
                print (phase.upper())
                print('-' * 10)

                running_loss_fnd = 0.0
                running_loss = 0.0 
                tpred = []
                tlabel = []

                if self.mode == "eann":
                    running_loss_event = 0.0
                    tpred_event = []
                    tlabel_event = []

                for batch in tqdm(self.dataloaders[phase]):
                    batch_data=batch
                    for k,v in batch_data.items():
                        batch_data[k]=v.cuda()
                    label = batch_data['label']

                    if self.mode == "eann":
                        label_event = batch_data['label_event']

                    with torch.set_grad_enabled(phase == 'train'):
                        if self.mode == "eann":

                            outputs, outputs_event,fea = self.model(**batch_data)
                            loss_fnd = self.criterion(outputs, label)
                            loss_event = self.criterion(outputs_event, label_event)
                            loss = loss_fnd + loss_event
                            _, preds = torch.max(outputs, 1)
                            _, preds_event = torch.max(outputs_event, 1)

                        else:

                            outputs, output_text, output_visual, output_audio, loss_sha, loss_spec, loss_distill = self.model(phase, self.modality_type, **batch_data)
                            _, preds = torch.max(outputs, 1)
                            loss_text = self.criterion(output_text, label)
                            loss_visual = self.criterion(output_visual, label)
                            loss_audio = self.criterion(output_audio, label)
                            loss_cls = loss_text + loss_visual + loss_audio
                            loss = loss_cls + self.lambda_1*(loss_sha + loss_spec) + self.lambda_2*loss_distill


                        if phase == 'train':

                            if epoch <= 4:
                               loss.backward()
                               self.optimizer.step()
                               self.optimizer.zero_grad()
                            else:
                                self.pcgrad = AGC(self.optimizer, reduction='sum')
                                loss = self.lambda_1*(loss_sha + loss_spec) + self.lambda_2*loss_distill
                                Gall = self.pcgrad.pc_backward(output_text, output_visual, output_audio, label, [loss_visual, loss_text, loss_audio], self.model)
                                self.optimizer.zero_grad()
                                loss.backwards()
                                idx = 0
                                for group in self.optimizer.param_groups:
                                    for p in group['params']:
                                        p.grad += Gall[idx]
                                        idx += 1
                                self.optimizer.step()
                                self.optimizer.zero_grad()


                    tlabel.extend(label.detach().cpu().numpy().tolist())
                    tpred.extend(preds.detach().cpu().numpy().tolist())
                    running_loss += loss.item() * label.size(0)

                    if self.mode == "eann":
                        tlabel_event.extend(label_event.detach().cpu().numpy().tolist())
                        tpred_event.extend(preds_event.detach().cpu().numpy().tolist())
                        running_loss_event += loss_event.item() * label_event.size(0)
                        running_loss_fnd += loss_fnd.item() * label.size(0)
                    
                epoch_loss = running_loss / len(self.dataloaders[phase].dataset)
                print('Loss: {:.4f} '.format(epoch_loss))
                results = metrics(tlabel, tpred)
                print (results)

                wandb.log({
                    f'Loss/{phase}': epoch_loss,
                    f'Acc/{phase}': results['acc'],
                    f'F1/{phase}': results['f1'],
                    f'Auc/{phase}':results['auc'],
                    f'Recall/{phase}': results['recall'],
                    f'Precision/{phase}': results['precision']
                }, step=epoch)

                self.writer.add_scalar('Loss/'+phase, epoch_loss, epoch+1)
                self.writer.add_scalar('Acc/'+phase, results['acc'], epoch+1)
                self.writer.add_scalar('F1/'+phase, results['f1'], epoch+1)

                if self.mode == "eann":
                    epoch_loss_fnd = running_loss_fnd / len(self.dataloaders[phase].dataset)
                    print('Loss_fnd: {:.4f} '.format(epoch_loss_fnd))
                    epoch_loss_event = running_loss_event / len(self.dataloaders[phase].dataset)
                    print('Loss_event: {:.4f} '.format(epoch_loss_event))
                    self.writer.add_scalar('Loss_fnd/'+phase, epoch_loss_fnd, epoch+1)
                    self.writer.add_scalar('Loss_event/'+phase, epoch_loss_event, epoch+1)
                
                if phase == 'test':
                    if results['acc'] > best_acc_test:
                        best_acc_test = results['acc']
                        best_model_wts_test = copy.deepcopy(self.model.state_dict())
                        best_epoch_test = epoch+1
                        if best_acc_test > self.save_threshold:
                            torch.save(self.model.state_dict(), self.save_param_path + "_test_epoch" + str(best_epoch_test) + "_{0:.4f}".format(best_acc_test))
                            print ("saved " + self.save_param_path + "_test_epoch" + str(best_epoch_test) + "_{0:.4f}".format(best_acc_test) )
                    else:
                        if epoch-best_epoch_test >= self.epoch_stop-1:
                            is_earlystop = True
                            print ("early stopping...")

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print("Best model on test: epoch" + str(best_epoch_test) + "_" + str(best_acc_test))

        if self.mode == "eann":
            print("Event: Best model on test: epoch" + str(best_epoch_test_event) + "_" + str(best_acc_test_event))

        self.model.load_state_dict(best_model_wts_test)

        return self.test(best_model_wts_test)


    def test(self, best_model_wts_test):
        since = time.time()

        self.model.cuda()
        self.model.eval()   

        pred = []
        label = []

        if self.mode == "eann":
            pred_event = []
            label_event = []

        for batch in tqdm(self.dataloaders['test']):
            with torch.no_grad(): 
                batch_data=batch
                for k,v in batch_data.items():
                    batch_data[k]=v.cuda()
                batch_label = batch_data['label']

                if self.mode == "eann":
                    batch_label_event = batch_data['label_event']
                    batch_outputs, batch_outputs_event, fea = self.model(**batch_data)
                    _, batch_preds_event = torch.max(batch_outputs_event, 1)

                    label_event.extend(batch_label_event.detach().cpu().numpy().tolist())
                    pred_event.extend(batch_preds_event.detach().cpu().numpy().tolist())
                else:

                    batch_outputs, fea, _ = self.model('test', self.modality_type, **batch_data)
                    outputs, output_text, output_visual, output_audio, loss_sha, loss_spec, loss_distill = self.model('test', self.modality_type, **batch_data)

                _, batch_preds = torch.max(outputs, 1)

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_preds.detach().cpu().numpy().tolist())


        print (get_confusionmatrix_fnd(np.array(pred), np.array(label)))
        print (metrics(label, pred))


        wandb.log({
            'Confusion Matrix': get_confusionmatrix_fnd(np.array(pred), np.array(label)),
            'Test Accuracy': metrics(label, pred)['acc']
        })


        if self.mode == "eann" and self.model_name != "FANVM":
            print ("event:")
            print (accuracy_score(np.array(label_event), np.array(pred_event)))

        return metrics(label, pred)

