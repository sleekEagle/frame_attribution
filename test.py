import shap
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 1. Data ──────────────────────────────────────────────────────────────
X, y = load_iris(return_X_y=True)
feature_names = ["sepal_len", "sepal_wid", "petal_len", "petal_wid"]

scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train)
X_test_t  = torch.tensor(X_test)
y_train_t = torch.tensor(y_train, dtype=torch.long)

# ── 2. Model ─────────────────────────────────────────────────────────────
class IrisNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 3)        # raw logits, 3 classes
        )
    def forward(self, x):
        return self.net(x)

model = IrisNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn   = nn.CrossEntropyLoss()

# ── 3. Train ──────────────────────────────────────────────────────────────
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    loss = loss_fn(model(X_train_t), y_train_t)
    loss.backward()
    optimizer.step()

model.eval()
acc = (model(X_test_t).argmax(dim=1) == torch.tensor(y_test)).float().mean()
print(f"Test accuracy: {acc:.2%}")

# ── 4. SHAP DeepExplainer ─────────────────────────────────────────────────
# Background: small random subset of training data (50 rows is typical)
background = X_train_t[:50]

explainer = shap.DeepExplainer(model, background)

# Explain 5 test instances
instances = X_test_t[:5]
shap_values = explainer.shap_values(instances)   
# shap_values: list of 3 arrays, one per class, each shape (5, 4)

# ── 5. Inspect results ────────────────────────────────────────────────────
class_names = ["setosa", "versicolor", "virginica"]

for i, cls in enumerate(class_names):
    print(f"\nSHAP values for class '{cls}' (5 instances × 4 features):")
    df = np.round(shap_values[i], 4)
    print(f"  {'':12s} " + "  ".join(f"{f:>12s}" for f in feature_names))
    for row_idx, row in enumerate(df):
        print(f"  instance {row_idx}:  " + "  ".join(f"{v:>12.4f}" for v in row))