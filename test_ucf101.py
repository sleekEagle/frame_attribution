import torch
import os
from glob import glob
from PIL import Image
import func

ucf101dm = func.UCF101_data_model()
model = ucf101dm.model
inference_loader = ucf101dm.inference_loader
inference_class_names = ucf101dm.inference_class_names
class_names = ucf101dm.inference_class_names
class_labels = {}
for k in class_names.keys():
    cls_name = class_names[k]
    class_labels[cls_name.lower()] = k


def load_jpg_ucf101(l, g, c, n, inference_class_names, transform):
    name = inference_class_names[l]
    dir = os.path.join(
        "C:\\Users\\lahir\\Downloads\\UCF101\\jpgs", name, "v_{}_g{}_c{}".format(name, str(g).zfill(2), str(c).zfill(2))
    )
    path = sorted(glob(dir + "/*"), key=func.numericalSort)

    target_path = path[n * 16 : (n + 1) * 16]
    if len(target_path) < 16:
        print("not exist")
        return False

    video = []
    for _p in target_path:
        video.append(transform(Image.open(_p)))

    return torch.stack(video)



# ucf101dm = func.UCF101_data_model()
# model = ucf101dm.model
# inference_loader = ucf101dm.inference_loader
# inference_class_names = ucf101dm.inference_class_names


'''
Acuracy : 0.8575338233022137
'''

def test():
    n_samples = 0
    n_correct = 0
    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        inputs, targets = batch
        cls = [class_labels[t[0].split('_')[1].lower()] for t in targets]
        with torch.inference_mode():
            pred = model(inputs)
            pred_cls = torch.argmax(pred,dim=1)
            n_samples += len(pred_cls)
            n_correct += ((pred_cls == torch.tensor(cls)).sum()).item()
    print(f'Acuracy : {n_correct/n_samples}')

'''
same test without using the dataloader directly
Acuracy : 0.8540840602696272
'''
def test_noloader():
    n_samples = 0
    n_correct = 0
    start_idx = 0
    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        inputs, targets = batch
        cls = [class_labels[t[0].split('_')[1].lower()] for t in targets]
        filename = batch[1][0][0]
        full_path = ucf101dm.construct_vid_path_from_full(filename)
        video = ucf101dm.load_jpg_ucf101(full_path, n=start_idx)
        with torch.inference_mode():  
            pred = model(video.permute(1,0,2,3)[None,:])
            pred_cls = torch.argmax(pred,dim=1)
            n_samples += len(pred_cls)
            n_correct += ((pred_cls == torch.tensor(cls[start_idx])).sum()).item()

    print(f'Acuracy : {n_correct/n_samples}')


def test_mask_noloader():
    n_samples = 0
    n_correct = 0
    start_idx = 0
    mask_path = r'C:\Users\lahir\Downloads\UCF101\analysis\masks'

    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        inputs, targets = batch
        cls = [class_labels[t[0].split('_')[1].lower()] for t in targets]
        filename = batch[1][0][0]
        full_path = ucf101dm.construct_vid_path_from_full(filename)
        video = ucf101dm.load_jpg_ucf101(full_path, n=start_idx)
        with torch.inference_mode():  
            pred = model(video.permute(1,0,2,3)[None,:])
            pred_cls = torch.argmax(pred,dim=1)
            n_samples += len(pred_cls)
            n_correct += ((pred_cls == torch.tensor(cls[start_idx])).sum()).item()

    print(f'Acuracy : {n_correct/n_samples}')


def get_video_frame_motion_importance(video):
    pred = model(video.unsqueeze(0))
    pred_cls = torch.argmax(pred)
    pred_l = pred[0,pred_cls]
    n_frames = video.size(1)
    logits_frame = []
    for n in range(1, n_frames):
        inputs = copy_paste_frame(video, n-1, n)
        pred_ = model(inputs.unsqueeze(0))
        pred_cls_ = torch.argmax(pred_)
        pred_l_ = pred_[0,pred_cls_]
        logits_frame.append(pred_l_.item())

    out = {
        'original_logit': pred_l.item(),
        'logits_frame': logits_frame
    }
    return out

if __name__ == '__main__':
    test()

