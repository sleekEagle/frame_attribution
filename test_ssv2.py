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
Accuracy = 72.24288397098354
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
    test_s2s()


