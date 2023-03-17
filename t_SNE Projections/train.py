import argparse
import os
import glob
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Import modules
from models import CrossEntropy, SiameseNet
from DeepFeatures import DeepFeatures
from tSNE.plot_tsne import plot_tSNE
from torchvision import models


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature visulaizer")

    parser.add_argument('--DATA_FOLDER', type=str, default='dataset', help="path to the dataset")
    parser.add_argument('--Outputs_FOLDER', type=str, default='Outputs', help="path to store images")
    parser.add_argument('--TB_FOLDER', type=str, default='Outputs_triplet', help="path to store Tensorboard")
    parser.add_argument("--EXPERIMENT_NAME", default="Font_resnet")
    parser.add_argument("--batch", type=int, default=64, help="batch sizes for each gpus")
    parser.add_argument("--pretrained_model", default="pre_trained_model/model_triplet.pth")

    args = parser.parse_args() 

    args.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    print("Device: ", args.device)
    print()

    # Create DATA_FOLDER directory
    DATA_FOLDER = os.path.join(args.DATA_FOLDER, "train") 
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(os.path.join(DATA_FOLDER))

    # Create TB_FOLDER directory
    TB_FOLDER = os.path.join(args.TB_FOLDER, "Tensorboard") 
    if not os.path.exists(TB_FOLDER):
        os.makedirs(os.path.join(TB_FOLDER))

    args.tSNE = os.path.join(args.Outputs_FOLDER, "tSNE") 
    if not os.path.exists(args.tSNE):
        os.makedirs(os.path.join(args.tSNE))

    # Creating dataloader
    mean, std = 0.1307, 0.3081
    tfms  = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((mean,), (std,))
            ])
    lcd = ImageFolder(DATA_FOLDER, tfms)
    args.output_k = len(lcd.classes)

    # Dataloader
    data_loader = DataLoader(lcd, batch_size=args.batch, shuffle=True)


    #Use a pretained model
    model = models.resnet18(pretrained=False).to(args.device)
    #resnet = resnet18(pretrained=False)

    head = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(model.fc.in_features, 1024)),
        ('added_relu1', nn.ReLU(inplace=True)),
        ('fc2', nn.Linear(1024, 512)),
        ('added_relu2', nn.ReLU(inplace=True)),
        ('fc3', nn.Linear(512, 100))
    ]))

    model.fc = head
    checkpoint = torch.load(args.pretrained_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.fc = Identity() # Remove the prediction head
    model.eval() # Setup for inferencing
    #print(model)
    
    # Initialize Tensorboard Logging Class
    DF = DeepFeatures(model = model, 
                  tensorboard_folder = TB_FOLDER, 
                  experiment_name= args.EXPERIMENT_NAME)

    DF.custom_TF_log(args, lcd.classes, data_loader)
    # Calling plot_tSNE
    # plot_tSNE(data_loader, model, args)

