import os
from glob import glob
import func
import json
from models.ssv2 import VJEPA2
from dataloaders import ssv2
import random

def replace_frames(video, src_idx, dst_idx_list):
    new_video = video.clone()
    src_frame = video[:,src_idx,:]
    new_video[:, dst_idx_list, :, :] = src_frame.unsqueeze(1).expand(-1, len(dst_idx_list), -1, -1)
    return new_video

'''
video : 3,T,H,W
'''
def group_frames(model, video, gt_idx, GRP_THRESHOLD):
    _,T,_,_ = video.size()

    #get original pred
    orig_stat = func.get_pred_stats(model, video)
    correct = True

    # if the inference is wrong, use the predicted class as the GT
    if orig_stat['cls'] != gt_idx:
        correct = False

    def get_best_frame(video, idx1, idx2, cls_idx):
        v_1_2 = replace_frames(video, idx1, [idx2])
        v_2_1 = replace_frames(video, idx2, [idx1])
        batch = [v_1_2, v_2_1]
        stat_1_2, stat_2_1 = func.get_pred_stats_batch(model, batch)

        stat_1_2['correct'] = stat_1_2['cls']==cls_idx
        stat_2_1['correct'] = stat_2_1['cls']==cls_idx

        ret = {}
        if stat_1_2['correct'] and not stat_2_1['correct']:
            ret['idx'] = idx1
            ret['stat'] = stat_1_2
        elif stat_2_1['correct'] and not stat_1_2['correct']:
            ret['idx'] = idx2
            ret['stat'] = stat_2_1
        elif stat_2_1['correct'] and stat_1_2['correct']:
            if stat_1_2['margin_5'] >= stat_2_1['margin_5']:
                ret['idx'] = idx1
                ret['stat'] = stat_1_2
            elif stat_1_2['margin_5'] < stat_2_1['margin_5']:
                ret['idx'] = idx2
                ret['stat'] = stat_2_1
        else: 
            return -1
        return ret

    def get_best_frame_list(video, idx1, idx1_list, idx2, cls_idx):
        idx1_list = idx1_list + [idx1]
        v_1_2 = replace_frames(video, idx1, [idx2])
        v_2_1 = replace_frames(video, idx2, idx1_list)

        batch = [v_1_2, v_2_1]
        stat_1_2, stat_2_1 = func.get_pred_stats_batch(model, batch)

        stat_1_2['correct'] = stat_1_2['cls']==cls_idx
        stat_2_1['correct'] = stat_2_1['cls']==cls_idx

        ret = {}
        if stat_1_2['correct'] and not stat_2_1['correct']:
            ret['idx'] = idx1
            ret['stat'] = stat_1_2
        elif stat_2_1['correct'] and not stat_1_2['correct']:
            ret['idx'] = idx2
            ret['stat'] = stat_2_1
        elif stat_2_1['correct'] and stat_1_2['correct']:
            if stat_1_2['margin_5'] >= stat_2_1['margin_5']:
                ret['idx'] = idx1
                ret['stat'] = stat_1_2
            elif stat_1_2['margin_5'] < stat_2_1['margin_5']:
                ret['idx'] = idx2
                ret['stat'] = stat_2_1
        else: 
            return -1
        return ret

    group_dict = {}
    vid = video.clone()
    i = 0
    final_src_idx = i
    d = {}
    while i<T:

        if i==T-1:
            final_src_idx = i
            d[final_src_idx] = []
            # print(f'{i}, one frames cluster. not change to logits')
            break
        j=i+1
        best = get_best_frame(vid, i, j, orig_stat['cls'])
        if best != -1:
            src_idx = best['idx']
            best_stat = best['stat']
            delta = func.get_stat_change(orig_stat, best_stat)
            min_change = delta['margin_5_change']
        else:
            min_change = 1000
            best_stat = -1

        final_src_idx = i
        final_dst_idx = []
        grp_stat_list= [best_stat]

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
            best = get_best_frame_list(v_, src_idx, dst_idxs, j, orig_stat['cls'])
            if best != -1:
                src_idx = best['idx']
                best_stat = best['stat']
                delta = func.get_stat_change(orig_stat, best_stat)
                min_change = delta['margin_5_change']
            else:
                min_change = 1000
                grp_stat_list.append(best_stat)
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
        
        d[final_src_idx] = {
            'frames': grp_values,
            'grp_stat_list': grp_stat_list
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
    s = func.get_pred_stats(model, v)
    delta = func.get_stat_change(orig_stat, s)


    group_dict['all_group_logit'] = s['max_logit']
    group_dict['all_group_change'] = delta
    group_dict['grp_pred_cls'] = s['cls']
    group_dict['gt_cls'] = gt_idx
    group_dict['original_stat'] = orig_stat
    group_dict['all_grp_stats'] = s
    group_dict['correct'] = correct

    return group_dict


def group_frames_loader_UCF101(out_dir, resume_path, GRP_THRESHOLD = 1e-3):

    if resume_path:
        fnames = []
        with open(resume_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                record = json.loads(line)
                fnames.append(record['filename'])

    out_path = os.path.join(out_dir, f'groups_{GRP_THRESHOLD}.jsonl')
    #****************************************************************************
    # the model and the data loader
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
        print(f'{idx/len(inference_loader)*100:.2f} % is done.', end='\r')
        # if idx==40: break
        inputs, targets = batch
        cls = [class_labels[t[0].split('_')[1].lower()] for t in targets]
        video = inputs[0,:]
        gt_idx = class_labels[targets[0][0].split('_')[1].lower()]
        filename = targets[0][0]

        # if filename != 'v_Surfing_g04_c01':
        #     continue
        if resume_path:
            if filename in fnames: continue

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

from torchvision import utils
from PIL import Image
from pathlib import Path

def save_video(out_path, video):
    os.makedirs(out_path, exist_ok=True)

    # Save each frame
    for frame_idx in range(video.shape[1]):
        # Extract single frame: shape [3, 256, 256]
        frame = video[:, frame_idx, :, :]
        
        # Convert to PIL Image (values assumed in [0, 1] or [0, 255])
        # If values are in [0, 1], multiply by 255 first
        frame = (frame - frame.min())/(frame.max()-frame.min())*255
        if frame.max() <= 1.0:
            frame = (frame * 255).byte()
        else:
            frame = frame.byte()
        
        # Convert to PIL and save
        frame_pil = Image.fromarray(frame.permute(1, 2, 0).cpu().numpy())
        frame_pil.save(f"{out_path}\\frame_{frame_idx:04d}.png")

def sample_paths_ssv2(N_SAMPLES=6):
    model = VJEPA2()
    model.eval()
    class_names = list(model.label2id.keys())

    d_names, paths = ssv2.get_ssv2_paths()

    # uniformly select and equal number of samples from a class
    unique_d = list(set(d_names))
    sel_d, sel_paths = [], []
    for d in unique_d:
        idxs = [i for i,s in enumerate(d_names) if s==d]
        random.shuffle(idxs)
        sel_idxs = idxs[:N_SAMPLES]
        sel_d.extend([d]*N_SAMPLES)
        sel_paths.extend([paths[i] for i in sel_idxs])

    # sanity check
    for i in range(len(sel_paths)):
        p = sel_paths[i]
        assert os.path.basename(os.path.dirname(p))==sel_d[i]
        
    #make sure all the class names are present in the list of dirs
    for d in sel_d:
        assert d in class_names, f'{d} is not in the list of dirs'

    out_path = 'dataloaders/ssv2_paths.txt'
    if os.path.exists(out_path):
        os.remove(out_path)
    with open(out_path, 'a') as f:
            for p in sel_paths:
                dir = os.path.basename(os.path.dirname(p))
                file = os.path.basename(p)
                s = dir + '/' + file
                f.write(s + '\n')

def group_frames_loader_SSV2(out_dir, GRP_THRESHOLD = 1e-3):
    out_path = os.path.join(out_dir, f'groups_{GRP_THRESHOLD}.jsonl')
    existing_list = []
    if os.path.exists(out_path):
        # os.remove(out_path)
        with open(out_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:  # Skip empty lines
                    d = json.loads(line)
                    existing_list.append(d['filename'])

    model = VJEPA2()
    model.eval()

    cls_list, path_list = ssv2.get_sampled_paths()
    n_files = len(path_list)
    for idx, p in enumerate(path_list):
        # print(f'{idx} of {n_files} is done.')
        print(f'{idx/n_files*100:.2f} is done',end='\r')
        # if not str(p) == 'C:\\Users\\lahir\\Downloads\\s2s_test\\Hitting something with something\\1337.webm':
        #     continue

        if p in existing_list: continue

        video = model.video_from_path(p)['pixel_values_videos'][0,:].permute(1,0,2,3)
        gt_idx = model.label2id[cls_list[idx]]
        group_dict = group_frames(model, video, gt_idx, GRP_THRESHOLD)
        if group_dict==-1:
            continue
        group_dict['filename'] = str(p)
        with open(out_path, 'a') as f:
            f.write(json.dumps(group_dict) + '\n')

        
if __name__ == '__main__':
    # sample_paths_ssv2()
    # out_dir = r'C:\Users\lahir\Downloads\UCF101\analysis\groups'
    out_dir = r'C:\Users\lahir\Downloads\ssv2_analysis'
    group_frames_loader_UCF101(out_dir, resume_path=r'C:\Users\lahir\Downloads\UCF101\analysis\groups\groups_0.001.jsonl', GRP_THRESHOLD=1e-3)
    # group_frames_loader_SSV2(out_dir, GRP_THRESHOLD=1e-3)

