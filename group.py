import torch
import os
from glob import glob
from PIL import Image
import func

GRP_THRESHOLD = 0.05

ucf101dm = func.UCF101_data_model()
model = ucf101dm.model
inference_loader = ucf101dm.inference_loader
inference_class_names = ucf101dm.inference_class_names
class_names = ucf101dm.inference_class_names
class_labels = {}
for k in class_names.keys():
    cls_name = class_names[k]
    class_labels[cls_name.lower()] = k


def replace_frame(video, src_idx, dst_idx):
    new_video = video.clone()
    new_video[:,dst_idx,:,:] = new_video[:,src_idx,:,:]
    return new_video

def replace_frames(video, src_idx, dst_idx_list):
    for dst in dst_idx_list:
        video = replace_frame(video, src_idx, dst)
    return video

def group_frames(model, video, gt_idx):
    _,T,_,_ = video.size()

    #get original pred
    stat = func.get_pred_stats(model, video)
    assert stat['pred_cls'] == gt_idx, 'original pred is not correct'

    def get_best_frame(video, idx1, idx2):
        v_1_2 = replace_frame(video, idx1, idx2)
        stat_1_2 = func.get_pred_stats(model, v_1_2, gt_idx, stat['pred_logit'])
        v_2_1 = replace_frame(video, idx2, idx1)
        stat_2_1 = func.get_pred_stats(model, v_2_1, gt_idx, stat['pred_logit'])
        change_list = [stat['per_change'] for stat in [stat_1_2,stat_2_1]]
        min_change = min(change_list)
        min_idx = change_list.index(min_change)
        v_ = [v_1_2, v_2_1][min_idx]
        best_idx = [idx1, idx2][min_idx]
        worst_idx = [idx1, idx2][1-min_idx]
        return v_, min_change, best_idx, worst_idx
    
    def get_best_frame_list(video, idx1, idx1_list, idx2):
        idx1_list = idx1_list + [idx1]
        v_1_2 = replace_frame(video, idx1, idx2)
        stat_1_2 = func.get_pred_stats(model, v_1_2, gt_idx, stat['pred_logit'])
        v_2_1 = replace_frames(video, idx2, idx1_list)
        stat_2_1 = func.get_pred_stats(model, v_2_1, gt_idx, stat['pred_logit'])
        change_list = [stat['per_change'] for stat in [stat_1_2,stat_2_1]]
        min_change = min(change_list)
        min_idx = change_list.index(min_change)
        v_ = [v_1_2, v_2_1][min_idx]
        best_idx = [idx1, idx2][min_idx]
        worst_idx = [idx1, idx2][1-min_idx]
        return v_, min_change, best_idx, worst_idx

    group_dict = {}
    i=0
    while i+1<T:

        j=i+1
        v_, min_change, src_idx, dst_idx = get_best_frame(video, i, j)

        dst_idxs = []
        while min_change < GRP_THRESHOLD:
            j+=1
            if j>=T:
                break
            final_src_idx = src_idx
            final_dst_idx = [idx for idx in list(range(i,j))]

            dst_idxs = [idx for idx in list(range(i,j)) if idx!=src_idx]
            v_ = replace_frames(video, src_idx, dst_idxs)
            s_ = func.get_pred_stats(model, v_, gt_idx, stat['pred_logit'])
            print(f'{src_idx} -> {dst_idxs} , {s_}')
            _, min_change, src_idx, dst_idx = get_best_frame_list(v_, src_idx, dst_idxs, j)
            print(min_change)
            pass

        i=max(final_dst_idx)+1
        group_dict[final_src_idx] = [idx for idx in final_dst_idx if idx!=final_src_idx]

    #test if the frames are replaced correctly
    # v = video.clone()
    # for src_idx, dst_idx_list in group_dict.items():
    #     v = replace_frames(v, src_idx, dst_idx_list)

    # for src_idx, dst_idx_list in group_dict.items():
    #     for d in dst_idx_list:
    #         print((v[:,src_idx,:,:] == v[:,d,:,:]).all())


    return group_dict


def group_frames_loader():
    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        inputs, targets = batch
        cls = [class_labels[t[0].split('_')[1].lower()] for t in targets]
        video = inputs[0,:]
        gt_idx = class_labels[targets[0][0].split('_')[1].lower()]
        group_dict = group_frames(model, video, gt_idx)

        print('*****************************************')
        stat = func.get_pred_stats(model, video)
        v = video.clone()
        for src_idx, dst_idx_list in group_dict.items():
            v = replace_frames(v, src_idx, dst_idx_list)
            s = func.get_pred_stats(model, v, gt_idx, stat['pred_logit'])
            print(f'{src_idx} -> {dst_idx_list} , {s}')
        
        # v = video.clone()
        # v = replace_frames(v, 4, [0,1,2,3,5,6,7,8,9])
        # print(func.get_pred_stats(model, v))
        
        
        pass

if __name__ == '__main__':
    group_frames_loader()