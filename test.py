import torch
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import json

from utils import AverageMeter

def test(data_loader, model, opt):
    print('test')

    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    start_accuracies = AverageMeter()
    end_accuracies = AverageMeter()
    mid_accuracies = AverageMeter()
    
    end_time = time.time()
    output_buffer = []
    test_results = {}
    previous_video_name = ''
    
    for i, (inputs, targets) in enumerate(data_loader):
        data_time.update(time.time() - end_time)
        
        with torch.no_grad():
            inputs = Variable(inputs)
            if not opt.no_cuda:
                inputs = inputs.cuda()
            outputs = model(inputs)
            assert len(outputs) == 3
            prob_s = outputs[0]
            prob_mid = outputs[1]
            prob_e = outputs[2]
            pred_s = prob_s.gt(0.5) #当概率大于0.5时，即判断为正样本
            pred_mid = prob_mid.gt(0.5)
            pred_e = pred_s.gt(0.t)
        for j in range(prob_s.size(0)):
            video_name = os.path.basename(targets['video_path'][j])
            frame_no = targets['frame_no'][j].item()
            annotation = targets['label'][j]
            assert len(annotation) == 3
            if annotation[0] != -3: #非无效groundtruth 标签
                if annotation[0] == pred_s[j].item():
                    start_accuracies.update(1,1)
                else:
                    start_accuracies.update(0,1)
                if annotation[1] == pred_mid[j].item():
                    mid_accuracies.update(1,1)
                else:
                    mid_accuracies.update(0,1)
                if annotation[2] == pred_e[j].item():
                    end_accuracies.update(1,1)
                else:
                    end_accuracies.update(0,1)
            if video_name != previous_video_name:
                if len(output_buffer) > 0:
                    test_results[previous_video_name] = output_buffer
                    output_buffer = []
            output_buffer.append((frame_no,prob_s[j].item(),prob_mid[j].item(),prob_e[j].item()))
        
        if (i % 100) == 0:
            with open(os.path.join(opt.result_path, 'predict.json'), 'w') as f:
                json.dump(test_results, f)

        batch_time.update(time.time() - end_time)
        end_time = time.time()
        
        print('[{}/{}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'.format(
                  i + 1,
                  len(data_loader),
                  batch_time=batch_time,
                  data_time=data_time))
    if previous_video_name != '':
        if len(output_buffer) > 0:
            test_results[previous_video_name] = output_buffer
            output_buffer = []
    print('Average accuracy of gradual start frame is %.3f',start_accuracies.avg)
    print('Average accuracy of gradual mid frame is %.3f',mid_accuracies.avg)
    print('Average accuracy of gradual end frame is %.3f',end_accuracies.avg)
    with open(os.path.join(opt.result_path, 'predict.json'), 'w') as f:
        json.dump(test_results, f)
