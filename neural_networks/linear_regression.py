import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


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


torch.manual_seed(42) # set the random seed for reproducibility this basically makes sure that the random numbers are the same every time you run the code, the random numbers are used for initializing the weights and biases
n_features = xtr.shape[1]
w = torch.randn(n_features, 1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

lr = 0.4
n_epochs = 20



print("Without pytorch high level api")
for e in range(n_epochs):
    y_pred = xtr @ w + b 
    l = ((y_pred - ytr) ** 2).mean()
    l.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()

    print(f"Epoch {e+1}/{n_epochs}, Loss: {l.item():.4f}")



# now with pytorch high level api, i.e nn module

import torch.nn as nn

print("With pytorch high level api")

mse = nn.MSELoss()
model = nn.Linear(in_features=n_features, out_features=1)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

for e in range(n_epochs):
    y_pred = model(xtr)
    l = mse(y_pred, ytr)
    l.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {e+1}/{n_epochs}, Loss: {l.item():.4f}")

