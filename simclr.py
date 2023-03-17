import torch
import utils
import os
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict


class SimCLR:
    def __init__(self, model, optimizer, dataloaders, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.loss_fn = loss_fn

    def load_model(self, args):
        path = os.path.join(args.model_path, "model.pth")
        checkpoint = torch.load(path)
        optim = torch.load("outputs/model/optimizer.pth")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(optim['optimizer_state_dict'])
        print("Model Loaded from: ", path)
        # for param in self.model.parameters():
        #         param.requires_grad = False
        # print(self.model)
        if 'remove_top_layers' in vars(args):
            if args.remove_top_layers > 0:
                if args.multiple_gpus:
                    temp = list(self.model.module.fc.children())
                    if args.remove_top_layers <= len(temp):
                        self.model.module.fc = torch.nn.Sequential(
                            *temp[:-len(temp)])
                else:
                    temp = list(self.model.fc.children())
                    print("Lenght of top layers", len(temp))
                    if args.remove_top_layers <= len(temp):
                        self.model.fc = torch.nn.Sequential(
                            *temp[:-len(temp)])
                        print("Top Layers removed")
        
        return self.model

  
    def train(self, args, num_epochs):
        '''
        trains self.model on the train dataset for num_epochs
        and saves model and loss graph after log_interval
        number of epochs
        '''

        batch_losses = []
        running_loss = []

        def logging(current_epoch, loss):
            # Plot the training losses Graph and save it
            Path(os.path.join(args.respath, "plots")).mkdir(
                parents=True, exist_ok=True)
            file_name = 'training_losses_' + str(current_epoch)
            file_name = file_name+'.png'
            file_name = os.path.join('plots', file_name)
            utils.plotfuncs.plot_losses(
                loss, 'Training Losses', os.path.join(args.respath, file_name))

            Path(os.path.join(args.respath, "model")).mkdir(
                parents=True, exist_ok=True)

            # Store model and optimizer files
            torch.save({'model_state_dict': self.model.state_dict()}, os.path.join(
                args.respath, "model/model.pth"))
            torch.save({'optimizer_state_dict': self.optimizer.state_dict()}, os.path.join(
                args.respath, "model/optimizer.pth"))

            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(
                args.respath, "model_{}.pth".format(current_epoch)))

            np.savez(os.path.join(args.respath, "model/lossesfile"),
                     np.array(loss))

        self.model.train()
        print("Training Started!")
        self.model.to(args.device)
        # run a for loop for num_epochs
        for epoch in range(num_epochs):
            loop = tqdm(self.dataloaders['train'], leave=True)
            # run a for loop for each batch
            for x1,x2,x3,_ in loop:

                # zero out grads
                self.optimizer.zero_grad()

                x1 = x1.to(args.device)
                x2 = x2.to(args.device)
                x3 = x3.to(args.device)
                # get their outputs
                y1 = self.model(x1)
                y2 = self.model(x2)
                y3 = self.model(x3)

                # get loss value
                loss = self.loss_fn(y1, y2, y3)

                batch_losses.append(loss.cpu().data.item())
                running_loss.append(loss.cpu().data.item())
                # perform backprop on loss value to get gradient values
                loss.backward()
                # run the optimizer
                self.optimizer.step()

                # update progress
                # print(l)
                loop.set_description(f"Epoch [{epoch}/{num_epochs}]")
                loop.set_postfix(loss=sum(running_loss) /
                                 len(running_loss), refresh=True)
            logging(epoch, running_loss)
            running_loss = []
        logging(epoch, batch_losses)
