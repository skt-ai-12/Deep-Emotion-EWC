from __future__ import print_function
import argparse
import numpy  as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from tqdm import tqdm
from torch.autograd import Variable
from data_loaders import Plain_Dataset, eval_data_dataloader
from deep_emotion import Deep_Emotion
from generate_data import Generate_data
from torch.utils.tensorboard import SummaryWriter
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# def Train(epochs,train_loader,val_loader,criterion,optmizer,device, writer):
def Train(train_loaders,criterion,optmizer,device,epochs_per_task,batch_size,name):
    '''
    Training Loop
    '''
    print("===================================Start Training===================================")
    
    
    net.train()   
    batch_size =batch_size
    epochs_per_task =epochs_per_task
    name = name
    fisher_estimation_sample_size =1024
    consolidate = True
    cuda = True
    for idx , train_loader in enumerate(train_loaders,1): #enumerate(train_datasets, 1): #이부분은  task 별 데이터 정리한 후에 
        writer = SummaryWriter('runs/'+ str(idx))
        for epoch in range(1, epochs_per_task+1):
            train_loss = 0
            train_correct = 0
            data_stream = tqdm(enumerate(train_loader, 1))       
            for batch_index , (x,y) in data_stream:
                data_size = len(x)
                dataset_size = len(train_loader.dataset)
                dataset_batches = len(train_loader)
                # print("-------------------")
                # print(x.shape)
                # print(y)
                # print(data_size)
                # print(dataset_size)
                # print(dataset_batches)
                # print("------------------//-")
                
                x = Variable(x).cuda() if cuda else Variable(x)
                y = Variable(y).cuda() if cuda else Variable(y)

                optmizer.zero_grad()            
                outputs = net(x)
                ce_loss = criterion(outputs,y)
                ewc_loss = net.ewc_loss(cuda=cuda)
                loss = ce_loss + ewc_loss
                loss.backward()
                optmizer.step()
                train_loss += loss.item()
                _, predicted = torch.max(outputs,1)
                precision = (predicted == y).sum().float() / len(x)
                train_correct += precision
        
                data_stream.set_description((
                        '=> '
                        'task: {task}/{tasks} | '
                        'epoch: {epoch}/{epochs} | '
                        'progress: [{trained}/{total}] ({progress:.0f}%) | '
                        'prec: {prec:.4} | '
                        'loss => '
                        'ce: {ce_loss:.4} / '
                        'ewc: {ewc_loss:.4} / '
                        'total: {loss:.4}'
                    ).format(
                        task=idx,
                        tasks=len(train_loaders),
                        epoch=epoch,
                        epochs=epochs_per_task,
                        trained=batch_index*batch_size,
                        total=dataset_size,
                        progress=(100.*batch_index/dataset_batches),
                        prec=float(precision),
                        ce_loss=float(ce_loss),
                        ewc_loss=float(ewc_loss),
                        loss=float(loss),
                    ))
                
            # train_loss = train_loss/len(train_dataset)
            # train_acc = train_correct.double() / len(train_dataset)
            # print('Epoch: {} \tTraining Loss: {:.8f} \tTraining Acuuarcy {:.3f}% '
            #                                                     .format(epoch+1, train_loss,train_acc * 100))
            # writer.add_scalar("Loss/train", train_loss, epoch)
            # writer.add_scalar("Accuracy/train", train_acc, epoch)    
                    
        if consolidate and idx < 3:
                # estimate the fisher information of the parameters and consolidate
                # them in the network.
                print(
                    '=> Estimating diagonals of the fisher information matrix...',
                    flush=True, end='',
                )
                net.consolidate(net.estimate_fisher(
                    train_dataset, fisher_estimation_sample_size
                ))
                print(' Done!')      
    

    torch.save(net.state_dict(),'./pt/deep_emotion-{}-{}-{}-{}.pt'.format(name,epochs_per_task,batchsize,lr))
    print("===================================Training Finished===================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")    
    parser.add_argument('-d', '--data', type=str,required= True,
                               help='data folder that contains data files that downloaded from kaggle (train.csv and test.csv)')
    parser.add_argument('-e', '--epochs', type= int, required= True, help= 'number of epochs per task')
    parser.add_argument('-lr', '--learning_rate', type= float,required= True, help= 'value of learning rate')
    parser.add_argument('-bs', '--batch_size', type= int, required= True, help= 'training/validation batch size')
    parser.add_argument('-t', '--train', type=bool, required= True,help='True when training')
    parser.add_argument('-w', '--cweights', type=bool, help='True when class weighted')
    parser.add_argument('-n', '--name',type =str, required= True,help='name of pt' )
    args = parser.parse_args()
    
    
    epochspertask = args.epochs
    lr = args.learning_rate
    batchsize = args.batch_size
    name = args.name
    

    if args.train:
        net = Deep_Emotion()
        net.to(device)
        print("Model archticture: ", net)
        train_loaders = []
        for i in os.listdir(args.data):
            
            fol_name = i.split('.')[0]
            traincsv_file = args.data+'/'+i       
            # print(args.data[:-3])
            train_img_dir = args.data[:-3]+'/'+fol_name+'/'        
            
            transformation= transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
            train_dataset= Plain_Dataset(csv_file=traincsv_file, img_dir = train_img_dir, datatype = fol_name, transform = transformation)
            
            # validation_dataset= Plain_Dataset(csv_file=validationcsv_file, img_dir = validation_img_dir, datatype = 'val', transform = transformation)
            train_loader= DataLoader(train_dataset,batch_size=batchsize,shuffle = True,num_workers=0)
            # val_loader=   DataLoader(validation_dataset,batch_size=batchsize,shuffle = True,num_workers=0)
            train_loaders.append(train_loader)    
        
        
        cweights = [1.02660468, 9.40661861, 1.00104606, 0.56843877, 0.84912748, 1.29337298, 0.82603942]
        class_weights = torch.FloatTensor(cweights).cuda()
        if args.cweights:
          criterion= nn.CrossEntropyLoss(weight=class_weights)
        else:
          criterion= nn.CrossEntropyLoss()
        optmizer= optim.Adam(net.parameters(),lr= lr)
        # Train(epochs, train_loader, val_loader, criterion, optmizer, device, writer)
        Train( train_loaders, criterion, optmizer, device,epochspertask,batchsize,name)
