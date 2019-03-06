#-*- utf:8 -*-
import torch
import torch.utils.data as data
from PIL import Image
import os
import cv2

'''
获得视频片段
'''
def video_loader(video_path, center_frame_indices,sample_duration):
    video = []
    video_cap=cv2.VideoCapture(video_path)
    half_duration = sample_duration / 2 + sample_duration % 2
    frame_begin = int(center_frame_indices - half_duration + 1)
    repeat_head = 0
    frame_index = frame_begin
    if frame_begin < 0:
        repeat_head = abs(frame_begin)
        frame_index = 0
    video_cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
    for i in range(sample_duration - repeat_head):
        status,frame=video_cap.read()
        if status:
            frame=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert('RGB')
            if i == 0 and repeat_head > 0:
                for _ in range(repeat_head):
                    video.append(frame)
            video.append(frame)
        else:
            break
    if len(video) < sample_duration:
        video +=[video[-1] for _ in range(sample_duration - len(video))]
    return video
    
def get_default_video_loader():
    return video_loader
    
def make_dataset(root_path, sample_list_path, sample_duration):
    dataset = []
    with open(sample_list_path, 'r') as fd:
        for line in fd:
            words = line.strip().split(' ')
            video_name = words[0]
            frame_no = int(words[1])
            if len(words) < 5: #测试集
                label_s = -3
                label_mid = -3
                label_e = -3 #设置成一个无效值
            else:
                label_s = int(words[2])
                label_mid = int(words[3])
                label_e = int(words[4])
            label =(label_s,label_mid,label_e)
            sample={'video_path':os.path.join(root_path,video_name),
                    'frame_no':frame_no,
                    'label':label,
                    'sample_duration':sample_duration}
            assert(os.path.exists(sample['video_path']))
            dataset.append(sample)
    return dataset
    
class DataSet(data.Dataset):
    '''
    参数：
        root(string):视频文件所在的目录
        samples_list_path:样本说明文件的路径
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
    '''
    def __init__(self,root_path,samples_list_path,
                 spatial_transform=None, temporal_transform=None, target_transform=None,
                 sample_duration=21,get_loader=get_default_video_loader):
        
        self.data = make_dataset(root_path,samples_list_path,sample_duration)
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        self.target_transform = target_transform
        self.loader = get_loader()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (clip, target) where target is prob of the target class.
        """
        video_path = self.data[index]['video_path']
        center_index = self.data[index]['frame_no']
        sample_duration = self.data[index]['sample_duration']
        clip = self.loader(video_path,center_index,sample_duration)
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        target = self.data[index]
        #pdb.set_trace()
        if self.target_transform is not None:
            #pdb.set_trace()
            target = self.target_transform(target)
        return clip,target