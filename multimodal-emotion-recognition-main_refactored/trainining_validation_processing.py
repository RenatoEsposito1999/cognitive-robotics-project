from datasets.generate_dataset_RAVDESS import get_training_set_RAVDESS, get_validation_set_RAVDESS
from datasets.synchronized_data import Synchronized_data
from utils.logger import Logger
from train import train_epoch_multimodal
from validation import val_epoch_multimodal
from utils import transforms

import shutil
import os


import torch
from torch import nn, optim
from torch.optim import lr_scheduler
'''
    This method create the all necessary for doing the training and validation.
    
    Args:
        -opt = Object that contains all the argument
        -model = It is the model (multimodal transformer)
        -criterion_loss = Is the loss defined.
        -train_split_eeg = training set of EEG
        -validation_split_eeg = validation set of EEG
    
    Returns:
        None

'''
def training_validation_processing(opt, model, criterion_loss, train_split_eeg, validation_split_eeg):
    #Create the optimizer, the algorithm choosen is Adam
    optimizer = optim.SGD(
                model.parameters(),
                lr=opt.learning_rate,
                momentum=opt.momentum,
                dampening=opt.dampening,
                weight_decay=opt.weight_decay,
                nesterov=False)
    
    if not opt.no_train:
    
        video_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotate(),
                    transforms.ToTensor(opt.video_norm_value)])
        
        #Generate training set for audio and video
        training_data_audio_video = get_training_set_RAVDESS(opt, spatial_transform=video_transform) 
        
        #Generate training Loader for audio-video
        train_loader_audio_video = torch.utils.data.DataLoader(
            training_data_audio_video,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.n_threads,
            pin_memory=True)
        
        #This is used to pick randomly data synchronized with the batch of audio-video set, this because the number of data
        #of EEG is so smaller respect to video-audio data.
        EEGData_train = Synchronized_data(train_split_eeg)
        
        #Create two logger to save information about the training into a log file
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'prec1', 'lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'prec1', 'lr'])
        
        
    
        #This is the scheduler in order to have a dynamic learning rate    
        scheduler = lr_scheduler.StepLR(optimizer, 20, 0.1)
    
    if not opt.no_val:
    
        video_transform = transforms.Compose([
            transforms.ToTensor(opt.video_norm_value)])     
        
        #genereta validation set for audio-video data
        validation_data_audio_video = get_validation_set_RAVDESS(opt, spatial_transform=video_transform)
        
        #generate validation Loader for audio-video
        val_loader_audio_video = torch.utils.data.DataLoader(
            validation_data_audio_video,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        
        #This is used to pick randomly data synchronized with the batch of audio-video set, this because the number of data
        #of EEG is so smaller respect to video-audio data
        EEGData_val = Synchronized_data(validation_split_eeg)
           
        #Create a logger in order to save all information about the validation into a file log
        val_logger = Logger(
                os.path.join(opt.result_path, 'val.log'), ['epoch', 'loss', 'prec1'])
    
    best_prec1 = 0 #This variable is used to check if the current validation prec1 is better to the maximum obtained.
    
    if opt.resume_path:
        #This is used to resume the trained stopped, so import all the necessary
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']
        best_prec1 = checkpoint['best_prec1']
        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint["optimizer"])
                
    #Start training and validation
    print("Starto il training lets go: ")
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        
            if not opt.no_train:
                #This perform the training
                train_epoch_multimodal(i, train_loader_audio_video, model, criterion_loss, optimizer, opt,
                            train_logger, train_batch_logger, EEGData_train)
                scheduler.step() #Update the learning rate
                
                #Create a state, this contains the information that must be saved into pth file
                state = {
                    'epoch': i,
                    'arch': opt.arch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_prec1': best_prec1,
                    }
                 
                #Save pth
                save_checkpoint(state, model,  False, opt, train=True)
            
            if not opt.no_val:
                #Perform the validation step
                print('validation at epoch {}'.format(i))    
                loss_validation , prec1 = val_epoch_multimodal(EEGData_val, i, val_loader_audio_video, model, criterion_loss, opt,
                                            val_logger)
                
                #Check if the current precision of the validation is better than the maximum precision obtained until now
                is_best = prec1 > best_prec1
                best_prec1 = max(prec1, best_prec1)
                
            
                #Save the file pth    
                save_checkpoint(None, model, is_best, opt, train=False)



'''
    This function perform the saving of the information contained into state created by the training-validation process.
    This saves two files, in particular:
        1. The pth for each epoc related to the training
        2. The pth, if the prec1 of the current epoch validation is better than the maximum value
        (there are 4 save, because need to save the pth also for the machine without the gpu)
        
    Args:
        -State: dictionary that contains all the information that must be saved into the file pth
        -Model: This is used to save the pth in for machin without gpu
        -is_best: boolean for saving pth if prec1 is better than the maximum (For the validation)
        -opt: all the arguments 
        -Train: boolean, for check if the current state is related to the train or the validation
        
    returns:
        None

'''
def save_checkpoint(state, model, is_best, opt, train):
    if train:
        torch.save(state, '%s/%s_checkpoint'% (opt.result_path, opt.store_name)+'.pth')
        state["state_dict"]=model.module.state_dict()
        torch.save(state, '%s/%s_checkpoint'% (opt.result_path, opt.store_name)+'_cpu_.pth')
    if is_best:
        shutil.copyfile('%s/%s_checkpoint' % (opt.result_path, opt.store_name)+'.pth','%s/%s_best' % (opt.result_path, opt.store_name)+'.pth')
        shutil.copyfile('%s/%s_checkpoint' % (opt.result_path, opt.store_name)+'_cpu_.pth','%s/%s_best' % (opt.result_path, opt.store_name)+'_cpu_.pth')
    
    