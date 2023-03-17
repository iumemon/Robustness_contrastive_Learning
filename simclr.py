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
            # print("Adding new FC")
            # classifier = nn.Sequential(OrderedDict([
            #     ('fc1', nn.Linear(512, 1024)),
            #     ('added_relu1', nn.ReLU(inplace=True)),
            #     ('fc2', nn.Linear(1024, 512)),
            #     ('added_relu2', nn.ReLU(inplace=True)),
            #     # ('drop', nn.Dropout(p=0.5, inplace=False)),
            #     ('fc3', nn.Linear(512, 52))
            # ]))

            #self.model.fc = classifier
            # self.model.eval()
            # for param in self.model.parameters():
            #     param.requires_grad = False
            # net = torch.jit.script(model)
            # # model = torch.jit.freeze(net)
            # self.model.fc.fc1.weight.requires_grad = True
            # self.model.fc.fc1.bias.requires_grad = True
            # self.model.fc.fc2.weight.requires_grad = True
            # self.model.fc.fc2.bias.requires_grad = True
            # self.model.fc.fc3.weight.requires_grad = True
            # self.model.fc.fc3.bias.requires_grad = True

            # for name, grad in self.model.named_parameters():
            #     if grad.requires_grad:
            #         print(f"name: {name}")
        return self.model

    def get_representations(self, args, mode):

        self.model.eval()

        res = {
            'X': torch.FloatTensor(),
            'Y': torch.LongTensor()
        }

        with torch.no_grad():
            for batch, label in self.dataloaders[mode]:
                x = batch.to(args.device)

                # get their outputs
                pred = self.model(x)

                res['X'] = torch.cat((res['X'], pred.cpu()))
                res['Y'] = torch.cat((res['Y'], label.cpu()))

        res['X'] = np.array(res['X'])
        res['Y'] = np.array(res['Y'])

        return res

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
