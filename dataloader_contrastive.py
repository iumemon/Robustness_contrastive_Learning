import argparse
import torch
import torchvision
import utils
import simclr
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
import os
import random
from skimage import io
from os import walk
import matplotlib.pyplot as plt


random_transform = transforms.RandomApply(transforms=[
    transforms.RandomRotation(45, fill=255),
    transforms.RandomRotation(30, fill=255),
    transforms.RandomPerspective(distortion_scale=0.6, p=1.0, fill=255)], p=0)



class TrainDataset(torch.utils.data.Dataset):

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

    def augmented_image(self, image):
        return random_transform(image)

    def get_image_path(self, index):
        path = os.path.join(
            self.args.datapath, 'train', index)
        return path


    def get_positive_img(self, img_name):
        
        file_name = img_name
        name, file_extension = os.path.splitext(file_name)
        img_name = name + file_extension
        style_no = name.split('_')[0]
        char_no = name.split('_')[1]
        # Get random character
        char = random.randint(1,52)
        while char == char_no:
            char = random.randint(1,52)
        style_img_name = str(style_no)+"_" + str(char)+file_extension
        return style_img_name


    def __getitem__(self, index):


        image_name = self.filenames[index]
        image_path = self.get_image_path(image_name)
        anchor = io.imread(image_path)

        positive_image_name = self.get_positive_img(self.filenames[index])
        positive_image_path = self.get_image_path(positive_image_name)

        positive_image = io.imread(positive_image_path)
        #print("anchor: ", image_name)
        #print("positive_image: ", positive_image_name)

        f_name = self.filenames[index]
        s_label, c_label = f_name.split('_')#[1]
        c_label = c_label.split('.')[0]
        #print("label: ", c_label, " Style: ", s_label)
        x1 = self.tensorify(self.augmented_image(Image.fromarray(anchor)))
        x2 = self.tensorify(self.augmented_image(Image.fromarray(positive_image)))

        y_label = torch.tensor(int(s_label))
        return x1,x2, y_label



if __name__ == '__main__':
    args = parser.parse_args()
    args.device = torch.device('cuda' if args.cuda else 'cpu')


    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        TrainDataset(args),
        batch_size=args.batch_size,
        shuffle=True,
    )


    def show():

        images1, images2, labels = next(iter(dataloaders['train']))
        fig, axs = plt.subplots(2, 2)
        axs[0,0].imshow(images1[0][0], cmap = 'gray')
        axs[0,1].imshow(images1[0][1], cmap = 'gray')
        axs[1,0].imshow(images2[0][0], cmap = 'gray')
        axs[0,1].imshow(images2[0][1], cmap = 'gray')
        #print(labels[0].item())
        #print(labels[1].item())
        
        #     plt.subplot(1, 4, i+1)
        #     plt.imshow(images1[i][0], cmap = 'gray')
        #     print(labels[i].item())
        # for i in range(4):
        #     plt.subplot(1, 4, i+1)
        #     plt.imshow(images2[i][0],  cmap='gray')
        #     print(labels[i].item())

        plt.show()
    #show()
