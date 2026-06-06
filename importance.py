from models.ssv2 import VJEPA2
import shap
import numpy as np
import func
import json
import torch    
import os
from pathlib import Path
import CONST
from dataloaders import ssv2
import time

# ucf101dm = func.UCF101_data_model()
# model = ucf101dm.model

# def get_video():
#     with open(UCF_PATH, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue

#             record = json.loads(line)
#             filename = record['filename']
#             p = ucf101dm.construct_vid_path_from_full(filename)
#             video = ucf101dm.load_jpg_ucf101(p, n=0)
#             g = record['groups']
#             groups = {}
#             for k in g:
#                 if 'frames' in g[k]:
#                     f = g[k]['frames']
#                 else: 
#                     f = []
#                 groups[int(k)] = f
#             return video, groups


def batch_pred(model, t):
    pred = torch.empty(0).to(t.device)
    split_tensors = torch.split(t, split_size_or_sections=32, dim=0)
    for val in split_tensors:
        with torch.no_grad():
            p = model(val)
            pred = torch.concat([pred,p],axis=0)
    return pred
        
class CalcSHAP:
    def __init__(self, model, fill_method):
        self.model = model.to('cuda')
        self.fill_method = fill_method
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.model = self.model.to(self.device) 
        self.model.eval()
        # self.avg_pred = CONST.UCF_AVG_PRED
        self.avg_pred = CONST.SSV2_AVG_PRED
        self.n_masks = 0
        self.difference = 0

    def predict_with_mask(self, mask):
        self.n_masks += len(mask)
        preds = torch.empty(0).to('cuda')
        zero_idx = [i for i,m in enumerate(mask) if sum(m)==0]
        non_zero_idx = [i for i,m in enumerate(mask) if i not in zero_idx]

        def chunk_list(lst, chunk_size):
            return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

        masked = self.video.clone()
        # for m in mask:
        #     g = func.past_fill_all(m, self.groups)
        #     vid_g = func.create_grouped_video(masked.permute(1,0,2,3), g)
        #     vid_g = vid_g.permute(1,0,2,3)[None,:]
        #     p = self.model(vid_g.permute(0,2,1,3,4).to('cuda'))
        #     # print(p[0,0].item())
        #     preds = torch.concat([preds,p],dim=0)
        # preds[0,:] = self.avg_pred
        # return preds.detach().cpu().numpy()
        
        batches = chunk_list(non_zero_idx,8)
        nz_pred = torch.empty(0).to('cuda')
        for i,nzidx in enumerate(batches):
            # print(f'Processing sample {(i+1)/len(batches)*100:.2f} %')
            #predict for non zero masks
            vid_t = torch.empty(0).to(self.device)
            for i in nzidx:
                masked = self.video.clone()
                if self.fill_method=='future':
                    g = [func.future_fill_all(mask[i], self.groups)]
                elif self.fill_method=='past':
                    g = [func.past_fill_all(mask[i], self.groups)]
                elif self.fill_method=='middle':
                    g = [func.hybrid_fill_all(mask[i], self.groups, 'middle')]
                elif self.fill_method=='random':
                    g = [func.hybrid_fill_all(mask[i], self.groups, 'random')]
                elif self.fill_method=='late':
                    g = [func.past_fill_all(mask[i], self.groups),
                         func.future_fill_all(mask[i], self.groups)]

                for g_ in g:
                    vid_g = func.create_grouped_video(masked.permute(1,0,2,3), g_)
                    vid_g = vid_g.permute(1,0,2,3)[None,:]
                    vid_t = torch.concat([vid_t,vid_g])

            with torch.no_grad():
                p = self.model(vid_t.permute(0,2,1,3,4))
                nz_pred = torch.concat([nz_pred,p],dim=0)

        if self.fill_method=='late':
            idxs = torch.linspace(0, nz_pred.size(0)-1, nz_pred.size(0), dtype=torch.int)
            even_idx = idxs[::2]
            odd_idx = idxs[1:][::2]
            nz_pred = (nz_pred[even_idx] + nz_pred[odd_idx])*0.5
        
        preds = torch.zeros(len(mask), nz_pred.size(1))
        for idx,i in enumerate(non_zero_idx):
            preds[i,:] = nz_pred[idx]
        for idx in zero_idx:
            preds[idx] = self.avg_pred
        
        
        # from torch.distributions import Normal
        
        # mean = 2e-4
        # std = 2e-5
        # dist = Normal(mean, std)
        # pred = dist.sample((8, 101))
        # pred[0,:] = self.avg_pred
        return preds.detach().cpu().numpy()


        # return preds.detach().numpy()

    def explain(self, video, groups, check=False):
        self.n_masks = 0
        self.groups = groups
        self.video = func.create_grouped_video(video.permute(1,0,2,3), groups).permute(1,0,2,3).to(self.device)
        NUM_GROUPS = len(groups)
        background = np.zeros((1, NUM_GROUPS))

        masker = shap.maskers.Independent(
            data=background,
            max_samples=2 ** NUM_GROUPS   # allow full exact enumeration
        )
        explainer = shap.Explainer(
            model=self.predict_with_mask,
            masker=masker,
            algorithm="exact"             # force exact Shapley computation
        )
        test_instance = np.ones((1, NUM_GROUPS)) 
        shap_values = explainer(test_instance)

        if check:
            sv = shap_values.values[0,:]
            bv = shap_values.base_values[0,:]
            p  = self.model(self.video.permute(1,0,2,3)[None,:])[0,:].cpu().detach().numpy()
            sv = np.sum(sv,axis=0)
            difference = abs(p - bv - sv).mean()
            
            # print('**************************************************')
            # print(f'Groups: {list(groups.keys())}')
            # print(f'Difference : {difference}')
            # print('**************************************************')
            self.difference = difference

        return shap_values
    

def calc_shap_UCF101(GRP_PATH, OUT_PATH, FILL_METHOD):
    #****************************************************************************
    # the model and the data loader
    #****************************************************************************
    ucf101dm = func.UCF101_data_model()
    model = ucf101dm.model
    model = model.to('cuda')
    # t=torch.zeros(8,3,16,112,12)
    # t = t.to('cuda')
    # model(t)


    inference_loader = ucf101dm.inference_loader
    inference_class_names = ucf101dm.inference_class_names
    class_names = ucf101dm.inference_class_names
    class_labels = {}
    for k in class_names.keys():
        cls_name = class_names[k]
        class_labels[cls_name.lower()] = k
    #****************************************************************************

    #*************************************************************************
    # initialize shap model 
    #*************************************************************************
    ex = CalcSHAP(model, fill_method=FILL_METHOD)

    #construct the out path for logging
    f = 'exactSHAP_'+ FILL_METHOD + '_' + Path(GRP_PATH).stem.split('_')[-1]+'.jsonl'
    # d = os.path.dirname(GRP_PATH)
    out_path = os.path.join(OUT_PATH,f)

    #read groups
    n=0
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            print(f'{n/line_count*100:.1f}% is done.', end='\r')
            n+=1

            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            filename = record['filename']

            p = ucf101dm.construct_vid_path_from_full(filename)
            video = ucf101dm.load_jpg_ucf101(p, n=0)
            g = record['groups']
            groups = {}
            for k in g:
                if 'frames' in g[k]:
                    f = g[k]['frames']
                else: 
                    f = []
                groups[int(k)] = f

            shap_values = ex.explain(video, groups, check=True)

            d = {}
            d['filename'] = filename
            d['shapley_values'] = shap_values.values.tolist()
            d['base_values'] = shap_values.base_values.tolist()
            d['difference'] = ex.difference
            d['n_masks'] = ex.n_masks
            d['groups'] = groups

            with open(out_path, 'a') as f:
                f.write(json.dumps(d) + '\n')


def calc_shap_ssv2(GRP_PATH, OUT_PATH, FILL_METHOD):
    model = VJEPA2()
    model.eval()
    class_names = list(model.label2id.keys())

    cls_list, path_list = ssv2.get_sampled_paths()
    n_files = len(path_list)
    nice_names = [Path(p).parent.name + '/' + Path(p).name for p in path_list]

    #*************************************************************************
    # initialize shap model 
    #*************************************************************************
    ex = CalcSHAP(model, fill_method=FILL_METHOD)

    #construct the out path for logging
    f = 'exactSHAP_'+ FILL_METHOD + '_' + Path(GRP_PATH).stem.split('_')[-1]+'.jsonl'
    # d = os.path.dirname(GRP_PATH)
    out_path = os.path.join(OUT_PATH,f)

    n=0
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        line_count = sum(1 for _ in enumerate(f))
    with open(GRP_PATH, 'r', encoding='utf-8') as f:
        start_time = time.time()
        for line in f:
            end_time = time.time()
            elapsed = end_time - start_time

            print(f'{n/line_count*100:.1f}% is done. Time passed: {elapsed:.2f}s', end='\r')
            n+=1

            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            
            # ignore if the grouping changed the prediction
            if record['grp_pred_cls'] != record['original_stat']['cls']: continue

            filename = record['filename']
            filename = filename.split('/')[-2] + '/' + filename.split('/')[-1]
            idx = nice_names.index(filename)
            p = path_list[idx]

            g = record['groups']
            groups = {}
            for k in g:
                if 'frames' in g[k]:
                    f = g[k]['frames']
                else: 
                    f = []
                groups[int(k)] = f

            video = model.video_from_path(p)['pixel_values_videos'][0,:]
            shap_values = ex.explain(video, groups, check=False)

            d = {}
            d['filename'] = filename
            d['shapley_values'] = shap_values.values.tolist()
            d['base_values'] = shap_values.base_values.tolist()
            d['difference'] = ex.difference
            d['n_masks'] = ex.n_masks
            d['groups'] = groups

            with open(out_path, 'a') as f:
                f.write(json.dumps(d) + '\n')

if __name__ == "__main__":
    GRP_PATH = r'C:\Users\lahir\Downloads\ssv2_analysis\groups\groups_0.0001.jsonl'
    OUT_PATH = r'C:\Users\lahir\Downloads\ssv2_analysis\shap'
    FILL_METHOD = 'future'
    calc_shap_ssv2(GRP_PATH, OUT_PATH, FILL_METHOD)