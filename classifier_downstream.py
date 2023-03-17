import argparse
import torch
import os
import torch.nn as nn
import utils
import simclr
import torchvision
from PIL import Image
import numpy as np
from sklearn.metrics import classification_report
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

# making a command line interface
parser = argparse.ArgumentParser(
    description="This is the command line interface for the linear evaluation model")

parser.add_argument('datapath', type=str,
                    help="Path to the data root folder which contains train and test folders")

parser.add_argument('model_path', type=str,
                    help="Path to the trained self-supervised model")

# parser.add_argument('respath', type=str,
#                    help="Path to the results where the evaluation metrics would be stored. ")

parser.add_argument('-bs', '--batch_size', default=64,
                    type=int, help="The batch size for evalsimuation")

parser.add_argument('-n', '--epochs', default=100, type=int,
                    help="Number of epochs for self supervised training")

parser.add_argument('-nw', '--num_workers', default=1,
                    type=int, help="The number of workers for loading data")

parser.add_argument('-c', '--cuda', action='store_true')

parser.add_argument('-resume', action='store_true')

parser.add_argument('-test', action='store_true')

parser.add_argument('--multiple_gpus', action='store_true')

parser.add_argument('--remove_top_layers', action='store_true')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, args):
        self.args = args
        self.filenames = list(os.listdir(
            os.path.join(self.args.datapath, 'train')))

    def __len__(self):
        return len(self.filenames)

    def tensorify(self, img):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        img = data_transform(img)
        return img

    def __getitem__(self, index):
        image_path = os.path.join(
            self.args.datapath, 'train', self.filenames[index])
        image = Image.open(image_path)
        f_name = self.filenames[index]
        label = f_name.split('_')[0]
        label = (int(label)-1)
        y_label = torch.tensor(label)
        image = self.tensorify(image)
        return image, y_label


def train(model, optimizer, epochs, dataloaders, path):

    PATH = path
    criterion = nn.CrossEntropyLoss()
    print("Training on: ", args.device)
    model.to(args.device)
    print("Training Started")
    if (not args.test):
        for epoch in range(epochs):
            train_loss = 0
            n_correct = 0
            n_samples = 0
            # training steps
            model.train()
            loop = tqdm(dataloaders['train'], leave=True)
            for i, (images, labels) in enumerate(loop, 0):

                #print("Batch ID: ", i)
                #print(f"Images: {images.shape} Labels: {len(labels)}")
                images = images.to(args.device)
                labels = labels.to(args.device)

                # Forward pass
                outputs = model(images)
                #print(f"Outputs: {outputs.shape} Labels: {labels.shape}")
                loss = criterion(outputs, labels)
                train_loss += loss.item()*images.size(0)

                # max returns (value ,index)
                _, predicted = torch.max(outputs, 1)
                n_samples += images.size(0)
                n_correct += (predicted == labels).sum().item()
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loop.set_description(f"Epoch [{epoch}/{epochs}]")
                loop.set_postfix(loss=loss.item(), refresh=True,
                                 n=n_correct, s=n_samples)
            acc = 100.0 * n_correct / n_samples
            print(f'Train Accuracy: {round(acc,2)} %')
            # # validation steps
            # model.eval()
            # n_correct = 0
            # n_samples = 0
            # valid_loss = 0
            # loop = tqdm(dataloaders['val'], leave=True)
            # for i, (images, labels) in enumerate(loop, 0):
            #     # moves tensors to GPU
            #     images = images.to(args.device)
            #     labels = labels.to(args.device)
            #     # forward pass
            #     output = model(images)
            #     # loss in batch
            #     # max returns (value ,index)
            #     #print(f"images{images.shape}, Outputs: {outputs.shape}")
            #     _, predicted = torch.max(outputs, 1)
            #     n_samples += images.size(0)
            #     n_correct += (predicted == labels).sum().item()
            #     vloss = criterion(output, labels)
            #     # update validation loss
            #     valid_loss += vloss.item()*images.size(0)
            #     loop.set_description(f"Validation Loop")
            #     loop.set_postfix(loss=vloss.item(), refresh=True)
            # val_acc = 100.0 * n_correct / n_samples
            # print(f'Val Accuracy: {round(val_acc,2)} %')
            
        print('Finished Training')
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, PATH)
        print('Saving New Model')
    print("Testing")
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        loop = tqdm(dataloaders['test'], leave=False)
        for images, labels in loop:
            images = images.to(args.device)
            labels = labels.to(args.device)
            # print("Images: ", images.shape)
            outputs = model(images)
            # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
        test_acc = 100.0 * n_correct / n_samples
        print(f'Test Accuracy: {test_acc} %')
    return model


if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cuda')
    print("Getting Model")
    model = utils.model.get_model(args)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
        weight_decay=1e-4
    )

    dataset = Dataset(args)

    test_size = int(len(dataset) * 0.10)
    if (args.test):
        test_size = len(dataset)
    print(f"Dataset Size: { len(dataset)}")
    train_set, test_set = torch.utils.data.random_split(
        dataset, [len(dataset)-test_size, test_size])

    # print(f"Train Size: { len(train_set)}")
    # print(f"Test Size: { len(test_set)}")
    # val_size = int(len(train_set) * 0.10)
    # train_set, val_set = torch.utils.data.random_split(
    #     train_set, [len(train_set) - val_size, val_size])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True
    )

    # dataloaders['val'] = torch.utils.data.DataLoader(
    #     val_set,
    #     batch_size=args.batch_size,
    #     shuffle=False,
    #     num_workers=args.num_workers,
    #     drop_last=True
    # )

    dataloaders['test'] = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=True
    )

    simclrobj = simclr.SimCLR(
        model,
        optimizer,
        dataloaders,
        None
    )
    model = simclrobj.load_model(args)
    # print(model)
    # print(model.fc)

    # model = torchvision.models.resnet18(pretrained = False, num_classes = 100)
    # optimizer = torch.optim.Adam(
    #     model.parameters(),
    #     lr=0.0001,
    #     weight_decay=1e-4
    # )

    # print(model)
    model.fc = nn.Linear(512, 50)
    for name, grad in model.named_parameters():
        if grad.requires_grad:
            if name == 'fc.weight' or name == 'fc.bias':
                continue
            grad.requires_grad = False    # print(model)

    head = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(512, 512)),
        ('added_relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(512, 256)),
        ('added_relu2', nn.ReLU(inplace=True)),
        ('fc3', nn.Linear(256, 100))
    ]))

    model.fc = head


    for name, grad in model.named_parameters():
        if grad.requires_grad:
            print(f"name: {name}")

    # print(model)

    def show():
        images, labels = next(iter(dataloaders['test']))
        for i in range(4):
            plt.subplot(1, 4, i+1)
            plt.imshow(images[i][0],  cmap='gray')
            print(labels[i].item())
        plt.show()
    #show()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
        weight_decay=1e-4
    )
    PATH = "./classifier_contrastive.pth"
    if(args.resume or args.test):
        print(f"Resuming Training for: {args.epochs} epoch")
        checkpoint = torch.load("classifier_contrastive.pth")
        model.load_state_dict(checkpoint['model_state_dict'])

    sim = train(model=model, optimizer=optimizer,
                epochs=args.epochs, dataloaders=dataloaders, path = PATH)
