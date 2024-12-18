'''
This code is based on https://github.com/okankop/Efficient-3DCNNs
'''
import torch
from torch.autograd import Variable
import time
from utils import AverageMeter, calculate_precision
from ContrastiveLearning import SupervisedContrastiveLoss

def val_epoch_multimodal(EEGDataLoader_val, epoch, data_loader, model, criterion, opt, logger,modality='both',dist=None):
    #for evaluation with single modality, specify which modality to keep and which distortion to apply for the other modaltiy:
    #'noise', 'addnoise' or 'zeros'. for paper procedure, with 'softhard' mask use 'zeros' for evaluation, with 'noise' use 'noise'
    print('validation at epoch {}'.format(epoch))
    assert modality in ['both', 'audio', 'video']    
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_avarage = AverageMeter()
    prec1_audio_video_avarage = AverageMeter()
    prec1_eeg_avarage = AverageMeter()
   
    contrastive_loss_fn = SupervisedContrastiveLoss(temperature=0.5)
    end_time = time.time()
    
    
    for i, (item1, item2) in enumerate(zip(data_loader, EEGDataLoader_val)):
      
        data_time.update(time.time() - end_time)
        
        inputs_audio, inputs_visual, targets = item1
        
        EEG_inputs, EEG_targets = item2
     
        targets = targets.to(opt.device)
        EEG_inputs = EEG_inputs.to(opt.device)
        EEG_targets = EEG_targets.to(opt.device)
      
        
        

        '''if modality == 'audio':
            print('Skipping video modality')
            if dist == 'noise':
                print('Evaluating with full noise')
                inputs_visual = torch.randn(inputs_visual.size())
            elif dist == 'addnoise': #opt.mask == -4:
                print('Evaluating with noise')
                inputs_visual = inputs_visual + (torch.mean(inputs_visual) + torch.std(inputs_visual)*torch.randn(inputs_visual.size()))
            elif dist == 'zeros':
                inputs_visual = torch.zeros(inputs_visual.size())
            else:
                print('UNKNOWN DIST!')
        elif modality == 'video':
            print('Skipping audio modality')
            if dist == 'noise':
                print('Evaluating with noise')
                inputs_audio = torch.randn(inputs_audio.size())
            elif dist == 'addnoise': #opt.mask == -4:
                print('Evaluating with added noise')
                inputs_audio = inputs_audio + (torch.mean(inputs_audio) + torch.std(inputs_audio)*torch.randn(inputs_audio.size()))

            elif dist == 'zeros':
                inputs_audio = torch.zeros(inputs_audio.size())'''
        
        inputs_visual = inputs_visual.permute(0,2,1,3,4)
        inputs_visual = inputs_visual.reshape(inputs_visual.shape[0]*inputs_visual.shape[1], inputs_visual.shape[2], inputs_visual.shape[3], inputs_visual.shape[4])
        
        with torch.no_grad():
            inputs_visual = Variable(inputs_visual)
            inputs_audio = Variable(inputs_audio)
            targets = Variable(targets)
            EEG_inputs = Variable(EEG_inputs)
            EEG_targets = Variable(EEG_targets)
            
        
        audio_embeddings, video_embeddings, EEG_embeddigs, logits_audio_video, logits_eeg, output  = model(inputs_audio, inputs_visual, EEG_inputs)
        
        loss_contrastive = contrastive_loss_fn(audio_embeddings, video_embeddings, EEG_embeddigs, targets, EEG_targets)
        
        
        loss_audio_video = criterion(logits_audio_video, targets)
        loss_eeg = criterion(logits_eeg, EEG_targets)
        
        total_loss = loss_audio_video + loss_eeg + loss_contrastive
        
         
        prec1_audio_video = calculate_precision(logits_audio_video.data, targets.data)
        prec1_eeg = calculate_precision(logits_eeg.data, EEG_targets.data)
       
        
        losses_avarage.update(total_loss.data, opt.batch_size)
        prec1_audio_video_avarage.update(prec1_audio_video, opt.batch_size)
        prec1_eeg_avarage.update(prec1_eeg, opt.batch_size)


        batch_time.update(time.time() - end_time)
        end_time = time.time()

        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.5f} ({batch_time.avg:.5f})\t'
              'Data {data_time.val:.5f} ({data_time.avg:.5f})\t'
              'Loss {loss}\t'
              'Prec@1_audio_video {prec1_audio_video_avarage.val:.5f} ({prec1_audio_video_avarage.avg:.5f})\t'
              'Prec@1_eeg {prec1_eeg_avarage.val:.5f} ({prec1_eeg_avarage.avg:.5f})\t'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=total_loss,
                  prec1_audio_video_avarage=prec1_audio_video_avarage,
                  prec1_eeg_avarage=prec1_eeg_avarage))

    logger.log({'epoch': epoch,
                'loss': losses_avarage.avg.item(),
                'prec1_audio_video': prec1_audio_video_avarage.avg.item(),
                'prec1_eeg': prec1_eeg_avarage.avg.item()})

    return losses_avarage.avg.item(), prec1_audio_video_avarage.avg.item(), prec1_eeg_avarage.avg.item()

    
