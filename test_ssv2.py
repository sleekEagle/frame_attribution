import torch
import numpy as np
from pathlib import Path
from models.ssv2 import VJEPA2
from dataloaders import ssv2
import torch.nn.functional as F

'''
Accuracy = 69.18 % 
'''

def make_inference(model, video, class_names):
    # video = ucf101dm.load_jpg_ucf101(vid_path)
    # video = video.unsqueeze(0).permute(0,2,1,3,4)
    pred = model(video)
    pred = F.softmax(pred,dim=1)
    pred_cls = torch.argmax(pred,dim=1).item()
    ret = {
        'pred_original_class': class_names[pred_cls],
        'pred_original_idx': pred_cls,
    }

    return ret

'''
on local:
Accuracy = 72.24288397098354
on kaggle GPU T4 x2:
Accuracy : 72.23183709540818
'''

def test_s2s():
    model = VJEPA2()
    model.eval()
    class_names = list(model.label2id.keys())

    d_names, paths = ssv2.get_ssv2_paths()
    n_files = len(paths)
    
    #make sure all the class names are present in the list of dirs
    for c in class_names:
        assert c in d_names , f'{c} is not in the list of dirs'
        pass

    n_correct = 0
    n_samples = 0

    for idx, p in enumerate(paths):
        if idx>0:
            print(f'{idx/n_files*100:.2f} % is done. Running acc: {n_correct/n_samples*100:.2f} %', end='\r')
        gt_idx = model.label2id[d_names[idx]]
        with torch.no_grad():
            pred_cls = model.predict_from_path(p)
            if pred_cls==gt_idx:
                n_correct += 1
            else:
                with open(r'C:\Users\lahir\Downloads\ssv2_analysis\ssv2_incorrect.txt', 'w') as file:
                    file.write(str(p))
        n_samples += 1
    print(f'Accuracy = {n_correct/n_samples*100} \%')

'''

Avg pred logits: tensor([[-1.0778,  0.6990,  0.0568, -1.1150, -0.6463,  0.0467,  0.5537, -1.2209,
         -0.3039, -0.0201, -0.2625, -0.3234,  0.4118,  0.7303, -0.8373,  0.7890,
          0.1248,  0.0849,  0.0299,  0.3142, -0.0405,  0.2057, -0.0393, -0.4108,
         -1.3873, -1.5425, -1.5284, -0.7279, -0.7907, -0.5817, -0.9605, -0.9368,
         -0.6052,  0.7530, -0.1993,  0.1298, -0.6322, -0.3060, -0.4806, -0.9759,
         -0.3011, -0.4477, -0.1674,  0.3445, -1.0356,  0.5258,  0.0740, -0.4303,
          0.2813, -0.5024, -1.0347, -0.6601, -0.5644, -1.4575, -1.6013, -0.4994,
         -0.6104, -0.9334, -1.0029,  0.1093, -1.5850, -0.0883,  0.0469, -0.2097,
         -0.3035, -0.7060, -0.2136, -0.2885, -0.1417, -0.5575, -0.4238, -0.1766,
         -0.0220, -0.8908, -0.1653,  0.2650, -1.0012, -0.4367, -0.8566, -0.6796,
          0.3743,  0.1068, -0.3793, -1.4657, -0.6237, -0.8892, -0.5433, -0.7574,
         -0.5675, -0.2265, -0.7567, -1.1096, -0.9761, -0.0449, -0.0027,  0.5000,
          0.5055,  0.2067, -0.0975, -0.1594,  0.2018,  0.4710,  0.6243, -0.5040,
          0.2351,  0.0018,  1.1253,  0.2021,  0.5955,  0.8241, -0.1276, -0.8251,
          1.2611, -0.2842, -0.2081, -0.9460, -0.4292, -0.2864, -0.2117, -0.4058,
         -0.2267, -0.4140,  0.2730, -0.4036, -0.8669,  0.7838,  0.2873,  0.0128,
          0.3754,  0.4862,  0.7661,  0.0942, -0.2596, -0.6352, -0.6121,  0.0174,
         -0.7820, -0.6476,  0.1716,  0.0350,  0.0330,  0.4949,  0.3223,  0.5452,
          0.1143,  0.6349, -0.9266, -0.4272,  0.1078, -1.4459, -1.0669,  0.1306,
         -0.4056, -1.5137, -0.5333, -0.0482, -0.7165, -0.6523, -0.4334, -0.7187,
         -0.0883,  0.4214, -0.8798, -1.0111, -0.2081, -0.8169, -0.6419, -0.9071,
         -0.6229, -1.2692,  0.4388, -0.3283, -1.1532,  0.1642]],
       device='cuda:0')

Accuracy = 67.43295019157088
'''

def test_s2s_sampled():
    model = VJEPA2()
    model.eval()
    class_names = list(model.label2id.keys())

    # d_names, paths = ssv2.get_ssv2_paths()
    # n_files = len(paths)

    cls_list, path_list = ssv2.get_sampled_paths()
    n_files = len(path_list)
    
    n_correct = 0
    n_samples = 0
    avg_pred = torch.zeros((1,174)).to('cuda')

    for idx, p in enumerate(path_list):
        if idx>0:
            print(f'{idx/n_files*100:.2f} % is done. Running acc: {n_correct/n_samples*100:.2f} %', end='\r')
        gt_idx = model.label2id[cls_list[idx]]
        with torch.no_grad():
            video = model.video_from_path(p)['pixel_values_videos'].permute(0,2,1,3,4)
            pred = model(video)
            avg_pred += pred
            pred_cls = torch.argmax(pred,dim=1).item()
            if pred_cls==gt_idx:
                n_correct += 1
        n_samples += 1
    print(f'Accuracy = {n_correct/n_samples*100} \%')
    print(f'Avg pred logits: {avg_pred/n_samples}')


def test_s2s_batch():
    model = VJEPA2()
    model.eval()
    class_names = list(model.label2id.keys())

    d_names, paths = ssv2.get_ssv2_paths()
    n_files = len(paths)
    
    #make sure all the class names are present in the list of dirs
    for c in class_names:
        assert c in d_names , f'{c} is not in the list of dirs'
        pass
    
    pred_t = model.predict_from_batch_path(paths)
    pass


if __name__ == '__main__':
    test_s2s_sampled()


