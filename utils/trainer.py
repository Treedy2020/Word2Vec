import os
import torch
import json
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer:
    def __init__(
        self,
        model,
        epoch,
        train_dataloader,
        val_dataloader,
        training_step,
        valid_step,
        criterion,
        optimizer,
        lr_scheduler,
        checkfrequency,
        model_savepath,
        checkpoint_path,
        loss_path,
        
    ):
        self.model = model
        self.model.to(device)
        self.epoch = epoch
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.training_step = training_step
        self.valid_step = valid_step
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.checkfrequency = checkfrequency
        self.model_savepath = model_savepath
        self.checkpoint_path = checkpoint_path
        
        self.loss = {
            'train':[],
            'val':[]
        }
        self.loss_path = loss_path
    
    def train(self):
        for e in range(self.epoch):
            self.train_epoch()
            self.valid_epoch()
            print('Training loss: {:5f}, Valid loss: {:5f}'.format(
                self.loss['train'][-1], 
                self.loss['val'][-1]
                )
            )
            
            self.lr_scheduler.step()
            if (not (e + 1)%self.checkfrequency) or ((e + 1) == self.epoch):
                self.getCheckpoint(e + 1)
        self.saveLoss()
        
    
    def train_epoch(self):
        self.model.train()
        running_loss = []
        
        for ind, data in enumerate(self.train_dataloader):
            src, tag = data[0].to(device), data[1].to(device)
            
            self.optimizer.zero_grad()
            out = self.model(src)
            loss = self.criterion(out, tag)
            loss.backward()
            running_loss.append(loss.item())
            self.optimizer.step()

            if ind == self.training_step:
                break
        
        epoch_loss = np.mean(running_loss)
        
        self.loss['train'].append(epoch_loss)

    def valid_epoch(self):
        self.model.eval()
        running_loss = []
        
        with torch.no_grad():
            for ind, data in enumerate(self.val_dataloader):
                src, tag = data[0].to(device), data[1].to(device)
                
                out = self.model(src)
                loss = self.criterion(out, tag)
                running_loss.append(loss.item())

                if ind == self.valid_step:
                    break
        
        epoch_loss = np.mean(running_loss)     
        self.loss['val'].append(epoch_loss)
    
    def getCheckpoint(self, epoch):
        save_path = os.path.join(self.model_savepath, 'model.pt')
        torch.save(self.model, save_path)
    
    def saveLoss(self):
        loss_path = os.path.join(self.loss_path, 'loss.json')
        with open(loss_path, 'w') as f:
            json.dump(self.loss, f) 

