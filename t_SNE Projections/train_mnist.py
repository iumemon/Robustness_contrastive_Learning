import argparse
import os
import glob
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Import modules
from models import CrossEntropy, Triplet
from DeepFeatures import DeepFeatures
from tSNE.plot_tsne import plot_tSNE
from torchvision import models
from torchvision.datasets import MNIST


class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature visulaizer")

    parser.add_argument('--DATA_FOLDER', type=str, default='dataset', help="path to the dataset")
    parser.add_argument('--IMGS_FOLDER', type=str, default='Outputs', help="path to store images")
    parser.add_argument('--EMBS_FOLDER', type=str, default='Outputs', help="path to store embeddings")
    parser.add_argument('--TB_FOLDER', type=str, default='Outputs', help="path to store Tensorboard")
    parser.add_argument("--EXPERIMENT_NAME", default="Font_resnet")
    parser.add_argument("--batch", type=int, default=256, help="batch sizes for each gpus")
    parser.add_argument("--pretrained_model", default="pre_trained_model/CE_model/FONT_PAIR_LOSS_2500_32_0.0005_5_1.0.pth")

    args = parser.parse_args() 

    args.device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    print("Device chosen is ", args.device)
    print()

    # Create DATA_FOLDER directory
    DATA_FOLDER = os.path.join(args.DATA_FOLDER, "train") 
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(os.path.join(DATA_FOLDER))

    # Create IMGS_FOLDER directory
    IMGS_FOLDER = os.path.join(args.IMGS_FOLDER, "Images") 
    if not os.path.exists(IMGS_FOLDER):
        os.makedirs(os.path.join(IMGS_FOLDER))

    # Create EMBS_FOLDER directory
    EMBS_FOLDER = os.path.join(args.EMBS_FOLDER, "Embeddings") 
    if not os.path.exists(EMBS_FOLDER):
        os.makedirs(os.path.join(EMBS_FOLDER))

    # Create TB_FOLDER directory
    TB_FOLDER = os.path.join(args.TB_FOLDER, "Tensorboard") 
    if not os.path.exists(TB_FOLDER):
        os.makedirs(os.path.join(TB_FOLDER))

    args.tSNE = os.path.join(args.IMGS_FOLDER, "tSNE") 
    if not os.path.exists(args.tSNE):
        os.makedirs(os.path.join(args.tSNE))


    from torch.utils.tensorboard import SummaryWriter

    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter('Outputs/Tensorboard/Font_resnet')

    def stack(tensor, times=3):
      return(torch.cat([tensor]*times, dim=0))

    tfs = transforms.Compose([transforms.Resize((221,221)), 
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485], std=[0.229]),
                              stack])

    trainset  = MNIST(root = DATA_FOLDER, download=True, transform=tfs)
    data_loader = torch.utils.data.DataLoader(trainset ,
                                              batch_size=args.batch,
                                              shuffle=True)

    # constant for classes
    classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

    # Initializing model
    ###### classifier ######
    # model = CrossEntropy(12).to(args.device)
    ##### Triplet net ######
    # model = Triplet(32).to(args.device)
    # model.load_state_dict(torch.load(args.pretrained_model))

    # model.model.fc = Identity() # Remove the prediction head
    # model.eval() # Setup for inferencing

    # Use a pretained model

    model = models.resnet152(pretrained=True).to(args.device)
    model.fc = Identity() # Remove the prediction head
    model.eval() # Setup for inferencing

    # helper function
    def select_n_random(data, labels, n=1500):
        '''
        Selects n random datapoints and their corresponding labels from a dataset
        '''
        assert len(data) == len(labels)

        perm = torch.randperm(len(data))
        return data[perm][:n], labels[perm][:n]

    # select random images and their target indices
    images, labels = select_n_random(trainset.data, trainset.targets)

    # get the class labels for each image
    class_labels = [classes[lab] for lab in labels]

    # log embeddings
    features = images.view(-1, 28 * 28)
    # writer.add_embedding(features,
    #                     metadata=class_labels,
    #                     label_img=images.unsqueeze(1))

    writer.add_embedding(features,
                        metadata=class_labels)
    writer.close()

    
    # # # # print(model)
    # # Initialize Tensorboard Logging Class
    # DF = DeepFeatures(model = model, 
    #               imgs_folder = IMGS_FOLDER, 
    #               embs_folder = EMBS_FOLDER, 
    #               tensorboard_folder = TB_FOLDER, 
    #               experiment_name= args.EXPERIMENT_NAME)

    # # Write Embeddings to Tensorboard
    # batch_imgs, batch_labels = next(iter(data_loader))
    # # get the class labels for each image for meta data
    # class_labels = [classes[lab] for lab in batch_labels]

    # DF.write_embeddings(x = batch_imgs.to(args.device))
    # DF.create_tensorboard_log(class_labels)

    # Calling plot_tSNE
    # plot_tSNE(data_loader, model, args)

