# -*- coding:utf-8 -*-
import torch
from torch import nn

from models import talnn

def generate_model(opt):
    assert opt.model in [
        'talnn'
    ]
    if opt.model == 'talnn':
        from model.talnn import get_fine_tuning_parameters
        model = talnn.TALNN(
                    sample_size=opt.sample_size,
                    sample_duration=opt.sample_duration)
    if not opt.no_cuda:
        model = model.cuda()
    
    model = nn.DataParallel(model,device_ids=None)    
    if opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        assert opt.arch == pretrain['arch']
            
        model.load_state_dict(pretrain['state_dict'])
            
        parameters = get_fine_tuning_parameters(model, opt.ft_begin_index)
    else:
        prameters = model.parameters()
       
    return model,parameters
    