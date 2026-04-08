import os
from glob import glob
import func
import json

def replace_frame(video, src_idx, dst_idx):
    new_video = video.clone()
    new_video[:,dst_idx,:,:] = new_video[:,src_idx,:,:]
    return new_video

def replace_frames(video, src_idx, dst_idx_list):
    for dst in dst_idx_list:
        video = replace_frame(video, src_idx, dst)
    return video

def group_frames(model, video, gt_idx, GRP_THRESHOLD):
    _,T,_,_ = video.size()

    #get original pred
    stat = func.get_pred_stats(model, video)

    # we do not consider when the prediction is not correct
    if stat['pred_cls'] != gt_idx:
        return -1

    def get_best_frame(video, idx1, idx2):
        v_1_2 = replace_frame(video, idx1, idx2)
        stat_1_2 = func.get_pred_stats(model, v_1_2, gt_idx, stat['pred_logit'])
        v_2_1 = replace_frame(video, idx2, idx1)
        stat_2_1 = func.get_pred_stats(model, v_2_1, gt_idx, stat['pred_logit'])
        change_list = [abs(stat['per_change']) for stat in [stat_1_2,stat_2_1]]
        min_change = min(change_list)
        min_idx = change_list.index(min_change)
        v_ = [v_1_2, v_2_1][min_idx]
        best_idx = [idx1, idx2][min_idx]
        worst_idx = [idx1, idx2][1-min_idx]
        best_logit = [stat['logit'] for stat in [stat_1_2,stat_2_1]][min_idx]
        return v_, min_change, best_logit, best_idx, worst_idx
    
    def get_best_frame_list(video, idx1, idx1_list, idx2):
        idx1_list = idx1_list + [idx1]
        v_1_2 = replace_frame(video, idx1, idx2)
        stat_1_2 = func.get_pred_stats(model, v_1_2, gt_idx, stat['pred_logit'])
        v_2_1 = replace_frames(video, idx2, idx1_list)
        stat_2_1 = func.get_pred_stats(model, v_2_1, gt_idx, stat['pred_logit'])
        change_list = [abs(stat['per_change']) for stat in [stat_1_2,stat_2_1]]
        min_change = min(change_list)
        min_idx = change_list.index(min_change)
        v_ = [v_1_2, v_2_1][min_idx]
        best_idx = [idx1, idx2][min_idx]
        worst_idx = [idx1, idx2][1-min_idx]
        best_logit = [stat['logit'] for stat in [stat_1_2,stat_2_1]][min_idx]
        return v_, min_change, best_logit, best_idx, worst_idx

    group_dict = {}
    group_dict['original_logit'] = stat['pred_logit']
    i=0
    vid = video.clone()
    final_src_idx = i
    d= {}
    while i<T:

        if i==T-1:
            final_src_idx = i
            d[final_src_idx] = []
            # print(f'{i}, one frames cluster. not change to logits')
            break
        j=i+1
        v_, min_change, best_logit, src_idx, dst_idx = get_best_frame(vid, i, j)
        final_src_idx = i
        final_dst_idx = []
        min_change_list = [min_change]

        dst_idxs = []
        grp = False
        while min_change < GRP_THRESHOLD:
            grp = True
            j+=1
            final_src_idx = src_idx
            final_dst_idx = [idx for idx in list(range(i,j))]

            #logging
            dst_idxs = [idx for idx in list(range(i,j)) if idx!=src_idx]
            # v_ = replace_frames(vid, src_idx, dst_idxs)
            # s_ = func.get_pred_stats(model, v_, gt_idx, stat['pred_logit'])
            # print(f'{src_idx} -> {dst_idxs} , {s_}')

            if j==T:
                break

            v_ = replace_frames(vid, src_idx, dst_idxs)
            _, min_change, best_logit, src_idx, dst_idx = get_best_frame_list(v_, src_idx, dst_idxs, j)
            min_change_list.append(min_change)
            # print(min_change)

        grp_values = [idx for idx in final_dst_idx if idx!=final_src_idx]

        #logging
        # if len(grp_values)==0:
        #     print(f'{i}, one frames cluster. not change to logits')

        # comment to reset video after each group is formed. 
        # vid = replace_frames(vid, final_src_idx, grp_values)

        if grp:
            i=max(final_dst_idx)+1
        else:
            i+=1
            best_logit = stat['pred_logit']
        
        d[final_src_idx] = {
            'frames': grp_values,
            'grp_logit': best_logit,
            'min_change_list': min_change_list
        }


    #test if the frames are replaced correctly
    # v = video.clone()
    # for src_idx, dst_idx_list in group_dict.items():
    #     v = replace_frames(v, src_idx, dst_idx_list)

    # for src_idx, dst_idx_list in group_dict.items():
    #     for d in dst_idx_list:
    #         print((v[:,src_idx,:,:] == v[:,d,:,:]).all())


    group_dict['groups'] = d

    #what if we make all groups at the same time
    v = video.clone()
    for src_idx, d_ in d.items():
        if 'frames' not in d_: continue
        dst_idx_list = d_['frames']
        v = replace_frames(v, src_idx, dst_idx_list)
    s = func.get_pred_stats(model, v, gt_idx, stat['pred_logit'])
    group_dict['all_group_logit'] = s['logit']
    group_dict['all_group_per_change'] = s['per_change']
    group_dict['grp_pred_cls'] = s['pred_cls']
    group_dict['gt_cls'] = gt_idx
    group_dict['orig_logits'] = stat['pred_logits']
    group_dict['grp_logits'] = s['pred_logits']
    
    return group_dict


def group_frames_loader_UCF101(GRP_THRESHOLD = 0.01):
    out_path = os.path.join(r'C:\Users\lahir\Downloads\UCF101\analysis', f'groups_{GRP_THRESHOLD}.jsonl')
    #****************************************************************************
    # data loader
    #****************************************************************************
    ucf101dm = func.UCF101_data_model()
    model = ucf101dm.model
    inference_loader = ucf101dm.inference_loader
    inference_class_names = ucf101dm.inference_class_names
    class_names = ucf101dm.inference_class_names
    class_labels = {}
    for k in class_names.keys():
        cls_name = class_names[k]
        class_labels[cls_name.lower()] = k
    #****************************************************************************

    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        inputs, targets = batch
        cls = [class_labels[t[0].split('_')[1].lower()] for t in targets]
        video = inputs[0,:]
        gt_idx = class_labels[targets[0][0].split('_')[1].lower()]
        filename = targets[0][0]
        # if filename != 'v_BabyCrawling_g01_c04':
        #     continue
        group_dict = group_frames(model, video, gt_idx, GRP_THRESHOLD)
        if group_dict==-1:
            continue

        # print('*****************************************')
        # print('***************testing*******************')
        # stat = func.get_pred_stats(model, video)
        # v = video.clone()
        # for src_idx, dst_idx_list_ in group_dict['groups'].items():
        #     v = video.clone()
        #     if type(dst_idx_list_)==list:
        #         print(f'{src_idx} -> {dst_idx_list_}')
        #         continue
        #     dst_idx_list = dst_idx_list_['frames']
        #     v = replace_frames(v, src_idx, dst_idx_list)
        #     s = func.get_pred_stats(model, v, gt_idx, stat['pred_logit'])
        #     print(f'{src_idx} -> {dst_idx_list} , {s}')
        # print('*****************************************')


        group_dict['filename'] = filename
        with open(out_path, 'a') as f:
            f.write(json.dumps(group_dict) + '\n')
        
if __name__ == '__main__':
    group_frames_loader_UCF101()