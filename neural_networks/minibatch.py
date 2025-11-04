import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

h = fetch_california_housing()

device = 'cpu'
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Device: ", device)



# First split: separate test set (80% train+val, 20% test)
X_temp, xtt, y_temp, ytt = train_test_split(h.data, h.target, test_size=0.2, random_state=42)
# Second split: separate validation set (75% train, 25% val of the remaining 80%)
xtr, xva, ytr, yva = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

xtr = torch.FloatTensor(xtr)
xva = torch.FloatTensor(xva)
xtt = torch.FloatTensor(xtt)

ytr = torch.FloatTensor(ytr).reshape(-1, 1)
yva = torch.FloatTensor(yva).reshape(-1, 1)
ytt = torch.FloatTensor(ytt).reshape(-1, 1)

means = xtr.mean(dim=0, keepdim=True)
stds = xtr.std(dim=0, keepdim=True)

xtr = (xtr - means) / stds
xva = (xva - means) / stds
xtt = (xtt - means) / stds

train_dataset = TensorDataset(xtr, ytr)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(xva, yva)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

torch.manual_seed(42)
n_features = xtr.shape[1]
model = nn.Sequential(
    nn.Linear(n_features, 50),
    nn.ReLU(),
    nn.Linear(50, 40),
    nn.ReLU(),
    nn.Linear(40, 1),
)
model.to(device)

lr = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
mse = nn.MSELoss()

def train(model, optimizer, loss, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            l = loss(y_pred, y)
            total_loss += l.item()
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

        mean_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {mean_loss:.4f}")


def evaluate(model, data_loader, metric_fn, aggregate_fn=torch.mean):
    model.eval()
    metrics = []
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            metrics.append(metric_fn(y_pred, y))
    return aggregate_fn(torch.stack(metrics))



train(model, optimizer, mse, train_loader, epochs=20)
val_loss = evaluate(model, val_loader, mse)
print(f"Validation Loss: {val_loss:.4f}")









