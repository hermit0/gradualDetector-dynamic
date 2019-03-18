# -*- coding:utf-8 -*-
import torch
from torch import nn

from models import talnn,resnet_talnn

def generate_model(opt):
    assert opt.model in [
        'talnn','resnet_talnn'
    ]
    if opt.model == 'talnn':
        from models.talnn import get_fine_tuning_parameters
        model = talnn.TALNN(
                    sample_size=opt.sample_size,
                    sample_duration=opt.sample_duration)
    elif opt.mode = 'resnet_talnn':
        assert opt.model_depth in [10, 18, 34, 50, 101, 152, 200]
        from models.resnet_talnn import get_fine_tuning_parameters
        if opt.model_depth == 10:
            model = resnet_talnn.resnet_talnn10(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 18:
            model = resnet_talnn.resnet_talnn18(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 34:
            model = resnet_talnn.resnet_talnn34(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 50:
            model = resnet_talnn.resnet_talnn50(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 101:
            model = resnet_talnn.resnet_talnn101(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 152:
            model = resnet_talnn.resnet_talnn152(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
        elif opt.model_depth == 200:
            model = resnet_talnn.resnet_talnn200(
                shortcut_type=opt.resnet_shortcut,
                sample_size=opt.sample_size,
                sample_duration=opt.sample_duration)
    
    
    if not opt.no_cuda:
        model = model.cuda()
    
    model = nn.DataParallel(model,device_ids=None)    
    if opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        if "resnet_talnn" in opt.arch and "resnet" in pretrain['arch']:
            #可以用resnet的模型来初始化resnet_talnn
            resnet_depth = int(pretrain['arch'].split('-')[1])
            assert opt.model_depth == resnet_depth
            model.load_state_dict(pretrain['state_dict'],strict=False)
            parameters = model.parameters()
        else:    
            assert opt.arch == pretrain['arch']
            
            model.load_state_dict(pretrain['state_dict'])
            
            parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
    else:
        parameters = model.parameters()
       
    return model,parameters
    
