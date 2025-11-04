import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch.nn as nn


h = fetch_california_housing()

# First split: separate test set (80% train+val, 20% test)
X_temp, xtt, y_temp, ytt = train_test_split(h.data, h.target, test_size=0.2, random_state=42)
# Second split: separate validation set (75% train, 25% val of the remaining 80%)
xtr, xva, ytr, yva = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

xtr = torch.FloatTensor(xtr)
xva = torch.FloatTensor(xva)
xtt = torch.FloatTensor(xtt)

means = xtr.mean(dim=0, keepdims=True)
stds = xtr.std(dim=0, keepdims=True)

xtr = (xtr - means) / stds
xva = (xva - means) / stds
xtt = (xtt - means) / stds


ytr = torch.FloatTensor(ytr).reshape(-1, 1)
yva = torch.FloatTensor(yva).reshape(-1, 1)
ytt = torch.FloatTensor(ytt).reshape(-1, 1)


torch.manual_seed(42)
n_features = xtr.shape[1]
w = torch.randn(n_features, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.1
n_epochs = 20

model = nn.Sequential(
    nn.Linear(in_features=n_features, out_features=50),
    nn.ReLU(),
    nn.Linear(in_features=50, out_features=40),
    nn.ReLU(),
    nn.Linear(in_features=40, out_features=1),
)

optimizer = torch.optim.SGD(model.parameters(), lr=lr)
mse = nn.MSELoss()

for e in range(n_epochs):
    y_pred = model(xtr)
    l = mse(y_pred, ytr)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {e+1}/{n_epochs}, Loss: {l.item():.4f}")





