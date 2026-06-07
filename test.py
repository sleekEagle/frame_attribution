import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load data and train a model
X, y = load_iris(return_X_y=True, as_frame=False)
feature_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]

class CountedModel:
    def __init__(self, model):
        self.model = model
        self.call_count = 0

    def predict_proba(self, X):
        self.call_count += X.shape[0]   # count individual row evaluations
        return self.model.predict_proba(X)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
counted = CountedModel(model)

# Use a small background dataset (summary of training data)
background = X.mean(axis=0).reshape(1, -1)

explainer2 = shap.KernelExplainer(counted.predict_proba, background)
counted.call_count = 0   # reset before explain

instance = X[0:1]
shap_values2 = explainer2.shap_values(instance, nsamples=4)
print(f"Actual model calls recorded: {counted.call_count}")  # 320

print(f"maskMatrix shape  : {explainer2.maskMatrix.shape}")
print(f"Unique masks used : {explainer2.maskMatrix.shape[0]}")
print()
print("All masks:")
print(explainer2.maskMatrix.astype(int))

# explainer = shap.KernelExplainer(model.predict_proba, background)


# # shap_values = explainer.shap_values(instance, nsamples=32)  # <-- key parameter

# print("SHAP values per class:")
# for i, cls in enumerate(["setosa", "versicolor", "virginica"]):
#     print(f"  {cls}: {dict(zip(feature_names, shap_values[i][0].round(4)))}")