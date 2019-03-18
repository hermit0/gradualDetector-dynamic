#-*- coding:utf-8 -*-
'''
启动的主程序
'''
import os
import json
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from opts import parse_opts
from model import generate_model
from spatial_transforms import (Compose,
    ToTensor,Normalize,Scale,CenterCrop,RandomHorizontalFlip,MultiScaleCornerCrop)
from data.dataset import DataSet
from data import generate_train_samples
from utils import Logger
from target_transforms import ClassLabel
from train import train_epoch
import test

def get_mean(norm_value=255):
    return [
            114.7748 / norm_value, 107.7354 / norm_value, 99.4750 / norm_value
        ]
        
#获取最新的模型文件，将opt.resume_path设置为该文件的路径
def get_lastest_model(opt):
    if opt.resume_path != '':
        return
    epoch_num = -1
    for filename in os.listdir(opt.result_path):
        if filename[-3:]== 'pth':
            _epoch_num = int(filename[len('model_epoch'):-4])
            epoch_num = max(epoch_num,_epoch_num)
    if epoch_num > 0:
        opt.resume_path = os.path.join(opt.result_path,'model_epoch{}.pth'.format(epoch_num))

if __name__ == '__main__':
    opt = parse_opts()#解析命令行参数
    if opt.root_dir != '':
        opt.result_path = os.path.join(opt.root_dir, opt.result_path)
        if opt.train_list_path:
            opt.train_list_path = os.path.join(opt.root_dir,opt.train_list_path)
        if opt.train_gts_json_path:
            opt.train_gts_json_path  = os.path.join(opt.root_dir,opt.train_gts_json_path)
        if opt.test_list_path:
            opt.test_list_path = os.path.join(opt.root_dir,opt.test_list_path)
        if opt.resume_path:
            opt.resume_path = os.path.join(opt.root_dir, opt.resume_path)
        if opt.pretrain_path:
            opt.pretrain_path = os.path.join(opt.root_dir, opt.pretrain_path)
        opt.test_subdir = os.path.join(opt.root_dir,opt.test_subdir)
        opt.train_subdir = os.path.join(opt.root_dir,opt.train_subdir)
    opt.scales = [opt.initial_scale]
    for i in range(1, opt.n_scales):
        opt.scales.append(opt.scales[-1] * opt.scale_step)
    if opt.model == 'talnn':
        opt.arch = '{}'.format(opt.model)
    else:
        opt.arch = '{}-{}'.format(opt.model, opt.model_depth)#模型结构
    opt.mean = get_mean(opt.norm_value)
    if opt.auto_resume and opt.resume_path== '':
        get_lastest_model(opt)
    print(opt)
    with open(os.path.join(opt.result_path, 'opts.json'), 'w') as opt_file:
        json.dump(vars(opt), opt_file)
    
    torch.manual_seed(opt.manual_seed)
    
    model, parameters = generate_model(opt)
    print(model)
    criterion = nn.MSELoss()
    if not opt.no_cuda:
        criterion = criterion.cuda()
    norm_method = Normalize(opt.mean, [1, 1, 1])
    
    if not opt.no_train:
        crop_method = MultiScaleCornerCrop(opt.scales, opt.sample_size)
        spatial_transform = Compose([
            crop_method,
            RandomHorizontalFlip(),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = None
        target_transform = ClassLabel()
                        
        train_logger = Logger(
            os.path.join(opt.result_path, 'train.log'),
            ['epoch', 'loss', 'label_s acc', 'label_mid acc','label_end acc','lr'])
        train_batch_logger = Logger(
            os.path.join(opt.result_path, 'train_batch.log'),
            ['epoch', 'batch', 'iter', 'loss', 'label_s acc','label_mid acc','label_end acc', 'lr'])
        
        if opt.nesterov:
            dampening = 0
        else:
            dampening = opt.dampening
        optimizer = optim.SGD(
            parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            dampening=dampening,
            weight_decay=opt.weight_decay,
            nesterov=opt.nesterov)
        step_scheduler = lr_scheduler.StepLR(optimizer,opt.lr_step,gamma=0.1)
    if opt.resume_path:
        print('loading checkpoint {}'.format(opt.resume_path))
        checkpoint = torch.load(opt.resume_path)
        assert opt.arch == checkpoint['arch']

        opt.begin_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if not opt.no_train:
            optimizer.load_state_dict(checkpoint['optimizer'])
            step_scheduler.load_state_dict(checkpoint['step_scheduler'])
    print('run')
    for i in range(opt.begin_epoch, opt.n_epochs + 1):
        if not opt.no_train:
            #每次重新生成该周期的训练集文件,每个渐变采样4帧，渐变前后各采样1帧，跳过紧接着渐变的2帧
            generate_train_samples.main(opt.train_gts_json_path,4,0.25,2,opt.train_list_path)
            training_data = DataSet(opt.train_subdir,opt.train_list_path,
                                spatial_transform=spatial_transform,
                                temporal_transform=temporal_transform,
                                target_transform=target_transform, sample_duration=opt.sample_duration)
            train_loader = torch.utils.data.DataLoader(
                                training_data,
                                batch_size=opt.batch_size,
                                shuffle=True,
                                num_workers=opt.n_threads,
                                pin_memory=True)
            step_scheduler.step()
            train_epoch(i, train_loader, model, criterion, optimizer, opt,
                        train_logger, train_batch_logger,step_scheduler)
    if opt.test:
        spatial_transform = Compose([
            Scale(opt.sample_size),
            CenterCrop(opt.sample_size),
            ToTensor(opt.norm_value), norm_method
        ])
        temporal_transform = None
        target_transform = None
        test_data = DataSet(opt.test_subdir,opt.test_list_path,
                            spatial_transform=spatial_transform,
                            temporal_transform=temporal_transform,
                            target_transform=target_transform, sample_duration=opt.sample_duration)
        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=opt.batch_size,
            shuffle=False,
            num_workers=opt.n_threads,
            pin_memory=True)
        test.test(test_loader, model, opt)
