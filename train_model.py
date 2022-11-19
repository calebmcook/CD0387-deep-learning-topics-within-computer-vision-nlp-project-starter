#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import json
import os
import argparse
import logging
import sys
import smdebug.pytorch as smd

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

#To account for truncated or corrupt images
#https://knowledge.udacity.com/questions/919040
#https://stackoverflow.com/questions/12984426/pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, criterion, hook, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    #Set the SMDebug hook for the validation phase
    hook.set_mode(smd.modes.EVAL)
    
    print("Testing Model on Whole Testing Dataset")
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    total_loss = running_loss / len(test_loader)
    total_acc = running_corrects.double() // len(test_loader)
    logger.info(f"Testing Accuracy: {total_acc}, Test set: Average loss: {total_loss}")
    

def train(model, train_loader, validation_loader, criterion, optimizer, hook, device):
    ''' 
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs=10
    best_loss=1e6
    image_dataset={'train': train_loader ,'valid':validation_loader}
    loss_counter=0
    
    #Set the SMDebug hook for the training phase
    hook.set_mode(smd.modes.TRAIN)
    
    for epoch in range(epochs):
        logger.info(f"Epoch: {epoch}")
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
          
            for inputs, labels in image_dataset[phase]:
                inputs=inputs.to(device)
                labels=labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                if phase=='train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)                    

            epoch_loss = running_loss // len(image_dataset[phase])
            epoch_acc = running_corrects / len(image_dataset[phase])
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1
            logger.info('{} loss: {:.4f}, acc: {:.4f}, best loss: {:.4f}'.format(phase, epoch_loss, epoch_acc,best_loss))
        if loss_counter==1:
            break
        if epoch==0:
            break
    return model

    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        #requires_grad = False in order to allow fine-tuning
        param.requires_grad = False   
    
    #sequential layer will have 133 outputs, the number of possible dog classifications
    model.fc = nn.Sequential(
                   nn.Linear(2048, 128),
                   nn.ReLU(inplace=True),
                   nn.Linear(128, 133))
    return model


def create_data_loaders(args):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    testing_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.ImageFolder(root=args.training_dir, transform=training_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    validationset = torchvision.datasets.ImageFolder(root=args.validation_dir, transform=testing_transform)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    
    testset = torchvision.datasets.ImageFolder(root=args.testing_dir, transform=testing_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn)
    
    return trainloader, validationloader, testloader



def collate_fn(batch):
    #used because the images are not of the same size.
    #See pytorch help thread here: 
    #https://discuss.pytorch.org/t/runtimeerror-stack-expects-each-tensor-to-be-equal-size-but-got-3-224-224-at-entry-0-and-3-224-336-at-entry-3/87211/23
    
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)  



def model_fn(model_dir):
    model = Net()
    with open(os.path.join(model_dir, "model.pth"), "rb") as f:
        model.load_state_dict(torch.load(f))
    return model



def save_model(model, model_dir):
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
    
    
def main(args):
    '''
    TODO: Initialize a model by calling the net function
    Reference: https://knowledge.udacity.com/questions/896915
    '''
    
    logger.info(f'Hyperparameters. LR: {args.lr}, Batch Size: {args.batch_size}')
   
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on Device {device}")
    
    model=net()
    model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss(ignore_index=133)
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    #Register the SMDebug hook to save output tensors.
    hook = smd.Hook.create_from_json_file()
    hook.register_hook(model)
    
    #create loaders
    trainloader, validationloader, testloader = create_data_loaders(args)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    #Pass the SMDebug hook to the train and test functions
    logger.info("Starting Model Training")
    model=train(model, trainloader, validationloader, loss_criterion, optimizer, hook, device)
    '''
    TODO: Test the model to see its accuracy
    '''
    logger.info("Testing Model")
    test(model, testloader, loss_criterion, hook, device)

    '''
    TODO: Save the trained model
    '''
    logger.info("Saving Model")
    save_model(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pth"))    
    

if __name__=='__main__':
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser=argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 1)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )

    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--training-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--validation-dir", type=str, default=os.environ["SM_CHANNEL_VALIDATION"])
    parser.add_argument("--testing-dir", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    
    args=parser.parse_args()
    
    main(args)