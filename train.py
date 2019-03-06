import torch
from torch.autograd import Variable
import time
import os
import sys

from utils import AverageMeter, calculate_accuracy
import pdb
def train_epoch(epoch, data_loader, model, criterion, optimizer, opt,
                epoch_logger, batch_logger,step_scheduler):
    print('train at epoch {}'.format(epoch))

    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    start_accuracies = AverageMeter()
    end_accuracies = AverageMeter()
    mid_accuracies = AverageMeter()
    end_time = time.time()
    
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        
        print('debuging')
        print(type(targets))
        print(targets)
        
        if not opt.no_cuda:
            targets = targets.cuda(async=True)
            inputs = inputs.cuda()
        inputs = Variable(inputs)
        targets = Variable(targets)
        
        outputs = model(inputs) #y为长度为3的list
        
        #将target划分成不同类别
        target_list=target.chunk(targets.size()[1],dim=1)
        assert targets.size()[1] == 3
        loss0 = criterion(outputs[0],target_list[0])
        loss1 = criterion(outputs[1],target_list[1])
        loss2 = criterion(outputs[2],target_list[2])
        loss = loss0+loss1+loss2
        losses.update(loss.item(), inputs.size(0))
        
        pdb.set_trace()
        print('debuging train_epoch accuracy')
        start_acc = calculate_accuracy(outputs[0], targets)
        start_accuracies.update(start_acc,inputs.size(0))
        mid_acc = calculate_accuracy(outputs[1], targets)
        mid_accuracies.update(mid_acc,inputs.size(0))
        end_acc = calculate_accuracy(outputs[2], targets)
        end_accuracies.update(end_acc,inputs.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end_time)
        end_time = time.time()

        batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(data_loader) + (i + 1),
            'loss': losses.val,
            'label_s acc': start_accuracies.val,
            'label_mid acc':mid_accuracies.val,
            'label_end acc': end_accuracies.val,
            'lr': optimizer.param_groups[-1]['lr']
        })
        print('Epoch: [{0}][{1}/{2}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Label_s Acc {sacc.val:.3f} ({sacc.avg:.3f})'
              'Label_mid Acc {macc.val:.3f} ({macc.avg:.3f})'
              'Label_e Acc {eacc.val:.3f} ({eacc.avg:.3f})'.format(
                  epoch,
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time,
                  loss=losses,
                  sacc=start_accuracies,
                  macc=mid_accuracies,
                  eacc=end_accuracies))
    epoch_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'label_s acc': start_accuracies.avg,
        'label_mid acc':mid_accuracies.avg,
        'label_end acc': end_accuracies.avg,
        'lr': optimizer.param_groups[-1]['lr']
    })
    if epoch % opt.checkpoint == 0:
        save_file_path = os.path.join(opt.result_path,
                                      'model_epoch{}.pth'.format(epoch))
        states = {
            'epoch': epoch + 1,
            'arch': opt.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'step_scheduler': step_scheduler.state_dict(),
        }
        torch.save(states, save_file_path)