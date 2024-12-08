'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils.average_meter import AverageMeter
from utils.precision import calculate_precision

'''
    This function perform the training for the i-th epoch.
    
    Args:
        -epoch: the i-th epoch
        -data_loader_audio_video: this is the data_loader for audio video
        -model: the model that want to train
        -criterion_loss: the loss
        -optimizer: the algortithm choosen to udate the weigths
        -opt: all the arguments
        -epoch_logger: save in file log the information about i-th epoch
        -batch_logger: save in file log the information about the batch
        -EEGData_train: is the structure for having the batch syncronized with the batch of audio-video
        
    Returns:
        None

'''
def train_epoch_multimodal(epoch, data_loader_audio_video, model, criterion, optimizer, opt,
                epoch_logger, batch_logger, EEGData_train):
    print('train at epoch {}'.format(epoch))

    model.train()

    #All the metrics used to compute the avarage of the loss and precision
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_avarage = AverageMeter()
    prec1_avarage = AverageMeter()
 
    end_time = time.time()
    
    
    
    for i, item1 in enumerate(data_loader_audio_video):
        data_time.update(time.time() - end_time)

        audio_inputs, visual_inputs, targets = item1
        
        eeg_inputs, mask_inputs = EEGData_train.generate_artificial_batch(targets) # tensor shape [batch_size,seq_len,features]
         
        visual_inputs = visual_inputs.permute(0,2,1,3,4)
        visual_inputs = visual_inputs.reshape(visual_inputs.shape[0]*visual_inputs.shape[1], visual_inputs.shape[2], visual_inputs.shape[3], visual_inputs.shape[4])
        
        targets = targets.to(opt.device)
        eeg_inputs = eeg_inputs.to(opt.device)
        mask_inputs = mask_inputs.to(opt.device)
        audio_inputs = Variable(audio_inputs)
        visual_inputs = Variable(visual_inputs)
        EEG_inputs=Variable(eeg_inputs)

        targets = Variable(targets)
        
        logits_output, aux_eeg_logits = model(audio_inputs, visual_inputs, EEG_inputs, opt.device)
       
        partial_loss = criterion(logits_output, targets)

        eeg_loss = criterion(aux_eeg_logits,targets)

        total_loss = partial_loss + 0.1 * eeg_loss
               
        prec1 = calculate_precision(logits_output.data, targets.data)
    
        losses_avarage.update(total_loss.data, opt.batch_size)
        prec1_avarage.update(prec1, opt.batch_size)
       
        optimizer.zero_grad() 
        total_loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader_audio_video) + (i + 1),
            'loss': losses_avarage.val.item(),
            'prec1': prec1_avarage.val.item(),
            'lr': optimizer.param_groups[0]['lr']
        })
        if i % 10 ==0:
            print('Epoch: [{0}][{1}/{2}]\t lr: {lr:.5f}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {prec1_avarage.val:.5f} ({prec1_avarage.avg:.5f})\t'.format(
                        epoch,
                        i,
                        len(data_loader_audio_video),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses_avarage,
                        prec1_avarage=prec1_avarage,
                        lr=optimizer.param_groups[0]['lr']))

    epoch_logger.log({
        'epoch': epoch,
        'loss': losses_avarage.avg.item(),
        'prec1': prec1_avarage.avg.item(),
        'lr': optimizer.param_groups[0]['lr']
    })
    
    


 

    
    
