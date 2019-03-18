#-*- coding:utf-8 -*-
'''
根据train.json中的渐变标注生成渐变用训练集
'''
import json
from numpy import random
import pdb

'''
主函数，
输入:
    gts_json_path -- str, groundtruth json 文件
    samples_in -- int, 单个渐变过程中的采样帧数
    rate -- float, 渐变过程前后的采样帧数和m的比值。即渐变过程前（后）采样rate*samples_in帧
    skip_frames -- int, 跳过紧贴着渐变过程的skip_frames帧进行采样
    out_file_path -- 生成的最终样本的文件的路径
'''
def main(gts_json_path,samples_in,rate,skip_frames,out_file_path):
    gts = json.load(open(gts_json_path))
    all_samples = []
    for video_name, annotation in gts.items():
        samples_in_the_video = core_process(annotation,samples_in,rate,skip_frames,out_file_path)
        for frame_no,prob_s,prob_mid,prob_e in samples_in_the_video:
            all_samples.append((video_name,frame_no,prob_s,prob_mid,prob_e))
    save_samples(all_samples,out_file_path)
    
'''
保存样本文件
'''
def save_samples(samples_list,out_file_path):
    fd = open(out_file_path,'w')
    for video_name,frame_no,prob_s,prob_mid,prob_e in samples_list:
        line = '{} {} {} {} {}\n'.format(video_name,frame_no,prob_s,prob_mid,prob_e)
        fd.write(line)
    fd.close()
    
'''
为单个视频生成训练用渐变样本集
返回的样本集中的样本表示为（frame_no,prob_s,prob_mid,prob_e）
'''
def core_process(annotation,samples_in,rate,skip_frames,out_file_path):
    total_frames = int(annotation['frame_num'])
    prob_s = [] #每一帧是渐变开始帧的概率
    prob_e = [] #每一帧是渐变结束帧的概率
    prob_mid = [] #每一帧是渐变过程的中间帧的概率
    for i in range(total_frames):
        prob_s.append(0)
        prob_e.append(0)
        prob_mid.append(0)
    #pdb.set_trace()
    all_shots = []
    shot_begin = 0 #镜头的开始帧
    shot_end = -1 #镜头的结束帧
    for begin,end in annotation['transitions']:
        if end - begin < 1:
            continue
        shot_end = begin
        all_shots.append((shot_begin,shot_end))
        shot_begin = end
    shot_end = total_frames - 1
    all_shots.append((shot_begin,shot_end))
    
    all_graduals = []
    trans_index = 0
    for begin, end in annotation['transitions']:
        if end - begin < 1:
            continue
        trans_index += 1
        if end - begin == 1:
            continue    #跳过切变
        if end - begin == 2:
            continue    #跳过长度为1的渐变
            
        gradual = (all_shots[trans_index-1][0],all_shots[trans_index-1][1],
                    all_shots[trans_index][0],all_shots[trans_index][1]) #(前一镜头,后一镜头)
        all_graduals.append(gradual)    
            
    
    for (begin,end) in annotation['transitions']:
        if end - begin <= 2:
            continue    #跳过切变，并忽略掉长度为1的渐变
        #整个渐变过程的区间为[begin+1,end-1]
        for frame_no in range(begin+1,end):
            if frame_no == begin + 1:
                #渐变开始帧
                prob_s[frame_no] = 1
            if frame_no == end - 1:
                prob_e[frame_no] = 1
            if frame_no > begin + 1 and frame_no + 1 < end:
                #渐变过程的中间帧
                prob_mid[frame_no] = 1
    
    #这儿可以添加对prob_s,prob_e,prob_mid进行平滑的操作
    
    samples_out = int(samples_in * rate)
    results = []#采样的样本集
    if len(all_graduals) == 0:
        #如果视频不含渐变
        for frame_no in sample_from_interval(0,total_frames-1,samples_out):
            results.append((frame_no,prob_s[frame_no],prob_mid[frame_no],prob_e[frame_no]))
    else:
        for (s1,e1,s2,e2) in all_graduals:
            #采样渐变过程中的帧
            for frame_no in sample_from_gradual(e1+1,s2-1,samples_in):
                results.append((frame_no,prob_s[frame_no],prob_mid[frame_no],prob_e[frame_no]))
                
            #采样渐变前的帧
            interval_len = 20 #采样的区间长度为21帧
            #渐变的前一镜头为[s1,e1],从[e1-skip_frames-20,e1-skip_frames]中随机采样sample_out帧
            interval_end = e1 - skip_frames
            interval_begin = interval_end - interval_len
            if interval_begin < s1:
                interval_begin = s1
            for frame_no in sample_from_interval(interval_begin,interval_end,samples_out):
                results.append((frame_no,prob_s[frame_no],prob_mid[frame_no],prob_e[frame_no]))
                
            #采样渐变后的帧
            #渐变的后一镜头为[s2,e2],从[s2+skip_frames,s2+skip_frames+20]中随机采样samples_out帧
            interval_begin = s2 + skip_frames
            interval_end = s2 + skip_frames + interval_len
            if interval_end > e2:
                interval_end = e2
            for frame_no in sample_from_interval(interval_begin,interval_end,samples_out):
                results.append((frame_no,prob_s[frame_no],prob_mid[frame_no],prob_e[frame_no]))
    #移除掉重复的采样帧
    results = list(set(results))
    results.sort()
    return results
    

'''
从渐变[s,e]中采样m帧，必须采样s,e这两帧，剩下的渐变中间帧随机采样
'''
def sample_from_gradual(s,e,total_sample):
    has_sampled = []
    frame_nos = []
    prob = []
    for x in range(s,e+1):
        frame_nos.append(x)
        prob.append(0)
    gradual_len = e - s + 1 #渐变的长度
    #设置采样的总帧数
    if gradual_len < 2:
        return has_sampled #跳过长度为1的渐变
    if gradual_len < total_sample:
        total_sample = gradual_len
    
    has_sampled.append(s)
    has_sampled.append(e)
    #mid = int((s + e) / 2)
    #if total_sample >= 3:
    #    has_sampled.append(mid)
    if total_sample > 2:
        remain = total_sample - 2
        for i in range(s+1,e):
            prob[i-s] = 1.0 / (gradual_len - 2)
        for frame in random.choice(frame_nos,size=remain,p=prob,replace=False):
            has_sampled.append(frame)
    return has_sampled
    
'''
从区间[s,e]中随机抽样k帧
'''
def sample_from_interval(s,e,k):
    if k > e - s + 1:
        k = e - s + 1
    all_values = []
    if k <= 0:
        return all_values
    for value in range(s,e+1):
        all_values.append(value)
    return random.choice(all_values,size=k,replace=False)
    
    
if __name__ == '__main__':
    gts_json_path = input('Enter the groundtruth json file: ')
    samples_in = input('the number you need sample in one gradual transition: ')
    samples_in = int(samples_in)
    rate = input('Enter the rate of samples out of gradual with respect to samples_in\n'
                'ie.samples_out = rate * samples_in: ')
    rate = float(rate)
    skip_frames = input('Enter the skip_frames to skip near the gradual: ')
    skip_frames = int(skip_frames)
    out_file_path = input('Enter the path of result sample file: ')
    main(gts_json_path,samples_in,rate,skip_frames,out_file_path)