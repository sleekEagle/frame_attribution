import shap
import numpy as np
import func

masks = 0

def predict(x):
    print('in predict')
    return np.random.rand(x.shape[0], 3)  # Dummy prediction function for binary classification

def custom_masker(mask, x):
    global masks
    masks += 1
    # in this simple example we just zero out the features we are masking
    return (x * mask).reshape(1, len(x))


# # Train a model (example with binary classification)
# X = np.random.rand(1, 5)

# explainer = shap.explainers.Exact(predict, custom_masker)
# shap_values = explainer(X)  # Explain the first sample

# # Get just the explanations for the positive class
# shap_values = shap_values[..., 1]

# print(f"Total masks used: {masks}")
# print(f"Expected maximum: {2**5} = 32 masks per sample")
# print(f"Per sample average: {masks / 1:.1f}")
# pass


def calc_shap():
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

    #*************************************************************************
    # initialize shap model 
    #*************************************************************************
    explainer = shap.explainers.Exact(predict, custom_masker)

    for idx, batch in enumerate(inference_loader):
        print(f'{idx/len(inference_loader)*100:.0f} % is done.', end='\r')
        if idx==40: break
        inputs, targets = batch
        cls = [class_labels[t[0].split('_')[1].lower()] for t in targets]
        video = inputs[0,:]
        gt_idx = class_labels[targets[0][0].split('_')[1].lower()]
        filename = targets[0][0]
        pass


if __name__ == "__main__":
    calc_shap()