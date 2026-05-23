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
Acuracy : 0.8540840602696272
Mean pred: tensor([-4.4024e-01, -4.0674e-02,  7.1152e-01, -3.1609e-01,  3.2655e-01,
         1.7831e-01, -5.8854e-02,  1.8917e-01, -2.1221e-01, -3.4419e-02,
        -5.6707e-01,  3.1817e-01, -5.1251e-01,  7.8728e-02,  3.8241e-01,
         4.4927e-01,  4.1014e-01, -4.8625e-01, -1.0066e+00, -5.5163e-01,
         3.9891e-01,  4.5372e-02,  3.8610e-01,  3.4739e-01,  9.7761e-02,
        -2.0676e-01, -9.5986e-02,  1.1389e-01,  6.2678e-01,  7.2580e-02,
        -7.7313e-01, -1.0116e+00, -1.3274e-02, -8.0600e-02,  2.9928e-01,
         8.2731e-02,  3.4950e-02,  1.0807e+00, -2.0324e-01, -3.5794e-01,
        -4.5873e-01, -7.5685e-01,  5.8560e-01, -4.4144e-01, -1.4889e-01,
        -3.4966e-01, -5.2623e-02,  7.0776e-01, -1.3836e+00,  4.2931e-02,
        -2.8714e-01,  7.1029e-01,  1.3096e-01,  3.8652e-01, -2.7135e-01,
         3.3671e-01,  2.8045e-01,  8.6642e-01,  2.7443e-01,  8.2717e-01,
         3.5757e-01,  7.5802e-01, -5.2487e-01, -4.5703e-01,  5.8566e-01,
        -2.6089e-02,  6.8762e-02,  2.0057e-02, -2.0009e-01,  1.8733e-03,
         4.6717e-01,  4.6903e-01, -1.2525e+00, -2.2765e-01,  8.1686e-01,
        -7.7576e-02, -1.9280e-01,  1.7929e-01,  1.3325e+00, -5.9062e-01,
        -1.5585e+00, -1.9866e+00, -1.3540e-01,  2.2269e-01, -3.5128e-01,
         3.8874e-01,  1.1447e+00, -6.0240e-01, -6.0129e-01,  5.1854e-01,
         3.8325e-01, -2.7351e-01,  2.6948e-01, -8.2457e-01,  5.1184e-01,
        -5.6957e-02,  2.5863e-01, -9.5546e-01,  7.0229e-01,  5.4221e-01,
         2.4085e-01])
'''

def test():
    n_samples = 0
    n_correct = 0
    start_idx = 0
    mean_pred = torch.zeros(101)
    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.2f} % is done.', end='\r')
        inputs, targets = batch
        cls = [class_labels[t[0].split('_')[1].lower()] for t in targets]
        with torch.inference_mode():
            pred = model(inputs[start_idx][None,:])
            mean_pred += pred[0]
            pred_cls = torch.argmax(pred,dim=1)
            n_samples += len(pred_cls)
            n_correct += ((pred_cls == torch.tensor(cls[start_idx])).sum()).item()
    print(f'Acuracy : {n_correct/n_samples}')
    print(f'Mean pred: {mean_pred/n_samples}')

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

