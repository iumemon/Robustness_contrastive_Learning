import torch
import numpy as np
import os
import cv2
from tensorboardX import SummaryWriter


try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

class DeepFeatures(torch.nn.Module):

    
    
    '''
    This class extracts, reads, and writes data embeddings using a pretrained deep neural network. Meant to work with 
    Tensorboard's Embedding Viewer (https://www.tensorflow.org/tensorboard/tensorboard_projector_plugin).
    When using with a 3 channel image input and a pretrained model from torchvision.models please use the 
    following pre-processing pipeline:
    
    transforms.Compose([transforms.Resize(imsize), 
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]) ## As per torchvision docs
    
    Args:
        model (nn.Module): A Pytorch model that returns an (B,1) embedding for a length B batched input
        tensorboard_folder (str): The folder path where the resulting Tensorboard log should be written to
        experiment_name (str): The name of the experiment to use as the log name
    
   

    '''
    

    def __init__(self, model,
                 tensorboard_folder,
                 experiment_name=None):
        
        super(DeepFeatures, self).__init__()
        
        self.model = model
        self.model.eval()
        
        self.tensorboard_folder = tensorboard_folder
        
        self.name = experiment_name
        
        self.writer = None
        
            
    def generate_embeddings(self, x):
        '''
        Generate embeddings for an input batched tensor
        
        Args:
            x (torch.Tensor) : A batched pytorch tensor
            
        Returns:
            (torch.Tensor): The output of self.model against x
        '''
        return(self.model(x))

    
    def _create_writer(self, name):
        '''
        Create a TensorboardX writer object given an experiment name and assigns it to self.writer
        
        Args:
            name (str): Optional, an experiment name for the writer, defaults to self.name
        
        Returns:
            (bool): True if writer was created succesfully
        
        '''
        
        if self.name is None:
            name = 'Experiment_' + str(np.random.random())
        else:
            name = self.name
        
        dir_name = os.path.join(self.tensorboard_folder, 
                                name)
        
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        
        else:
            print("Warning: logfile already exists")
            print("logging directory: " + str(dir_name))
        
        logdir = dir_name
        self.writer = SummaryWriter(logdir=logdir)
        return(True)

 
    def custom_TF_log(self, args, classes, val_loader):
        data = []
        class_labels = []
        all_images = []

        val_iter = iter(val_loader)

        for i in tqdm(range(len(val_loader))):
            x, y = next(val_iter)
            x = x.to(args.device)
            y = y.to(args.device)
            
            # Generate embeddings
            embs = self.generate_embeddings(x)

            for idx in range(len(embs.cpu().data.numpy())):
                data.append(embs.cpu().data.numpy()[idx])
                class_labels.append(y[idx].numpy())
                all_images.append(x[idx].numpy())

        ## Stack into tensors
        all_embeddings = torch.Tensor(data)
        class_labels = [classes[lab] for lab in class_labels]
        all_images = torch.Tensor(all_images)
        print("Final embedding shape", all_embeddings.shape)
        print()
        print("Final all_images shape", all_images.shape)
        print()

        if self.writer is None:
            self._create_writer(self.name)

        self.writer.add_embedding(all_embeddings, 
            metadata = class_labels, label_img = all_images)