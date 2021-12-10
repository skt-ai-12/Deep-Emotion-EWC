from __future__ import print_function
import argparse
import numpy  as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms

from data_loaders import Plain_Dataset, eval_data_dataloader
from deep_emotion import Deep_Emotion
from generate_data import Generate_data
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def Train(epochs,train_loader,val_loader,criterion,optmizer,device, writer):
def Train(epochs,train_loader,criterion,optmizer,device, writer,fol):
    '''
    Training Loop
    '''
    
    print("===================================Start Training===================================")
    for e in range(epochs):
        train_loss = 0
        validation_loss = 0
        train_correct = 0
        val_correct = 0
        # Train the model  #


        net.train()
        for data, labels in train_loader:

            # print("-------------------")
            # print(data.shape)
            # print(labels)
            # dataset_size = len(train_loader.dataset)
            # dataset_batches = len(train_loader)
            # print(dataset_size)
            # print(dataset_batches)
            # print("------------------//-")            

            data, labels = data.to(device), labels.to(device)
            optmizer.zero_grad()
            
            outputs = net(data)
            loss = criterion(outputs,labels)
            loss.backward()
            optmizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs,1)
            train_correct += torch.sum(preds == labels.data)
        
        #validate the model#
        # net.eval()
        # for data,labels in val_loader:
        #     data, labels = data.to(device), labels.to(device)
        #     val_outputs = net(data)
        #     val_loss = criterion(val_outputs, labels)
        #     validation_loss += val_loss.item()
        #     _, val_preds = torch.max(val_outputs,1)
        #     val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss/len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        # validation_loss =  validation_loss / len(validation_dataset)
        # val_acc = val_correct.double() / len(validation_dataset)
        # print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Acuuarcy {:.3f}% \tValidation Acuuarcy {:.3f}%'
        #                                                    .format(e+1, train_loss,validation_loss,train_acc * 100, val_acc*100))
        print('Epoch: {} \tTraining Loss: {:.8f} \tTraining Acuuarcy {:.3f}% '
                                                           .format(e+1, train_loss,train_acc * 100))
        writer.add_scalar("Loss/train", train_loss, e)
        writer.add_scalar("Accuracy/train", train_acc, e)
        # writer.add_scalar("Loss/val", validation_loss, e)
        # writer.add_scalar("Accuracy/val", val_acc, e)

    torch.save(net.state_dict(),'./pt_resume/deep_emotion-{}-{}-{}-{}.pt'.format(fol,epochs,batchsize,lr))
    print("===================================Training Finished===================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-s', '--setup', type=bool, help='setup the dataset for the first time')
    parser.add_argument('-d', '--data', type=str,required= True,
                               help='data folder that contains data files that downloaded from kaggle (train.csv and test.csv)')
    parser.add_argument('-re', '--resume', type=int,
                               help=' 1 when train continuously, 0 not continuously)')
    parser.add_argument('-e', '--epochs', type= int, help= 'number of epochs')
    parser.add_argument('-lr', '--learning_rate', type= float, help= 'value of learning rate')
    parser.add_argument('-bs', '--batch_size', type= int, help= 'training/validation batch size')
    parser.add_argument('-t', '--train', type=bool, help='True when training')
    parser.add_argument('-w', '--cweights', type=bool, help='True when class weighted')
    parser.add_argument('-pt', '--pt', type=str, help='dir of model')
    parser.add_argument('-fol', '--folder_name', type=str, help='dir of data')
    args = parser.parse_args()

    if args.setup :
        generate_dataset = Generate_data(args.data)        
        generate_dataset.save_images()
           
    
    epochs = args.epochs
    lr = args.learning_rate
    batchsize = args.batch_size    
    fol = args.folder_name
    if args.train:
        net = Deep_Emotion()
        if args.resume:
            print("resume!")
            pt = args.pt
            model_data = torch.load(pt)       
            net.load_state_dict(model_data,strict=False)

        net.to(device)
        print("Model archticture: ", net)      
        traincsv_file = args.data+'/'+fol +'.csv'        
        train_img_dir = args.data+'/'+fol +'/'       
        transformation= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
        train_dataset= Plain_Dataset(csv_file=traincsv_file, img_dir = train_img_dir, datatype = fol, transform = transformation)
        # validation_dataset= Plain_Dataset(csv_file=validationcsv_file, img_dir = validation_img_dir, datatype = 'val', transform = transformation)
        train_loader= DataLoader(train_dataset,batch_size=batchsize,shuffle = True,num_workers=0)
        # val_loader=   DataLoader(validation_dataset,batch_size=batchsize,shuffle = True,num_workers=0)
        writer = SummaryWriter('runs/fer2013_experiment_1')
        cweights = [1.02660468, 9.40661861, 1.00104606, 0.56843877, 0.84912748, 1.29337298, 0.82603942]
        class_weights = torch.FloatTensor(cweights).cuda()
        if args.cweights:
            criterion= nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion= nn.CrossEntropyLoss()
        optmizer= optim.Adam(net.parameters(),lr= lr)
        # Train(epochs, train_loader, val_loader, criterion, optmizer, device, writer)
        Train(epochs, train_loader, criterion, optmizer, device, writer,fol)
