#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

#References: See train_and_deploy.ipynb for full list, including from Common Model Architecture Types and Fine-Tuning lesson, finetune_a_cnn_solution.py

#TODO: Import dependencies for Debugging andd Profiling

def test(model, test_loader, criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
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
        running_corrects += torch.sum(preds == labels.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_corrects/ len(test_loader.dataset)
    print(f"Testing Accuracy: {100*total_acc}, Testing Loss: {total_loss}")
    
    

def train(model, train_loader, validation_loader, criterion, optimizer, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    epochs=1
    best_loss=1e6
    image_dataset={'train':train_loader, 'valid':validation_loader}
    loss_counter=0
    
    for epoch in range(epochs):
        for phase in ['train', 'valid']:
            print(f"Epoch {epoch}, Phase {phase}")
            if phase=='train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            running_samples=0

            for step, (inputs, labels) in enumerate(image_dataset[phase]):
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
                running_corrects += torch.sum(preds == labels.data).item()
                running_samples+=len(inputs)
                if running_samples % 2000  == 0:
                    accuracy = running_corrects/running_samples
                    print("Images [{}/{} ({:.0f}%)] Loss: {:.2f} Accuracy: {}/{} ({:.2f}%)".format(
                            running_samples,
                            len(image_dataset[phase].dataset),
                            100.0 * (running_samples / len(image_dataset[phase].dataset)),
                            loss.item(),
                            running_corrects,
                            running_samples,
                            100.0*accuracy,
                        )
                    )
                
                #NOTE: Comment lines below to train and test on whole dataset
                if running_samples>(0.2*len(image_dataset[phase].dataset)):
                    break

            epoch_loss = running_loss / running_samples
            epoch_acc = running_corrects / running_samples
            
            if phase=='valid':
                if epoch_loss<best_loss:
                    best_loss=epoch_loss
                else:
                    loss_counter+=1

        if loss_counter==1:
            break
    return model
    
    
def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    Student's note: Code adapted from exercise: "Common Model Architecture Types and Fine-Tuning", finetune_a_cnn_solution.py
    '''
    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        #requires_grad = False in order to allow fine-tuning
        param.requires_grad = False   

    num_features=model.fc.in_features
    
    #sequential layer will have 133 outputs, the number of possible dog classifications
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 133))
    return model


def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    TRAIN_DATASET_PATH="s3://sagemaker-us-east-1-260552509205/data/train/"
    TEST_DATASET_PATH="s3://sagemaker-us-east-1-260552509205/data/test/"
    VALID_DATASET_PATH="s3://sagemaker-us-east-1-260552509205/data/valid/"
    
    training_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    validation_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    testing_transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    trainset = torchvision.datasets.ImageFolder(root=TRAIN_DATASET_PATH, transform=training_transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
            shuffle=True)
    
    validationset = torchvision.datasets.ImageFolder(root=VALID_DATASET_PATH, transform=validation_transform)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size,
            shuffle=True)

    testset = torchvision.datasets.ImageFolder(root=TEST_DATASET_PATH, transform=testing_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
            shuffle=False)
    
    return trainloader, validationloader, testloader


def main(args):
    '''
    TODO: Initialize a model by calling the net function
    Reference: https://knowledge.udacity.com/questions/896915
    '''
    #batch_size=10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
    
    model=net()
    model.to(device)
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.0001)
    #hook = smd.Hook.create_from_json_file()
    #hook.register_hook(model)
    
    #create loaders
    trainloader, validationloader, testloader = create_data_loaders(batch_size=10)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    train(model, trainloader, validationloader, loss_criterion, optimizer, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, testloader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)
    
    

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    
    args=parser.parse_args()
    
    main(args)
