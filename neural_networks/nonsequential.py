import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
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

class WideandDeep(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_features, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(40 + n_features, 1)


    def forward(self, x):
        deep_output = self.deep_stack(x)
        wide_and_deep = torch.concat([x, deep_output], dim=1)
        return self.output_layer(wide_and_deep)


class WideandDeepV2(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_features, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(40 + n_features, 1)


    def forward(self, x):
        # breaking up the data into wide and deep parts
        x_wide = x[:, :5] 
        x_deep = x[:, 2:]

        # passing the deep part through the deep stack
        deep_output = self.deep_stack(x_deep)

        # concatenating the wide and deep parts
        wide_and_deep = torch.concat([x_wide, deep_output], dim=1)
        return self.output_layer(wide_and_deep)



# when dealing with multiple inputs (inputs may have different shapes or dimensions)

#option one for passing data
# 1. create and return wide and deep parts separately
# train_data_wide_and_deep = TensorDataset(xtr[:, :5], xtr[:, 2:], ytr)
# train_loader_wide_and_deep = DataLoader(train_data_wide_and_deep, batch_size=32, shuffle=True)

# updating training loop to adjust for the 3 tensors returned instead of the 2 tensors
# for xw, xd, yb in train_loader_wide_and_deep:
#     xw, xd, yb = xw.to(device), xd.to(device), yb.to(device)
#     y_pred = model(xw, xd)
#     l = loss(y_pred, yb)
#     total_loss += l.item()
#     l.backward()
#     optimizer.step()
#     optimizer.zero_grad()

#option two for passing data
#  create a classs

# class WideandDeepDataset(torch.utils.data.Dataset):
#     def __init__(self, xw, xd, y):
#         self.xw = xw
#         self.xd = xd
#         self.y = y

#     def __len__(self):
#         return len(self.xw)

#     def __getitem__(self, idx):
#         input_dict = {"wide": self.xw[idx], "deep": self.xd[idx]}
#         return input_dict, {"y": self.y[idx]}


# train_dataset_class = WideandDeepDataset(xtr[:, :5], xtr[:, 2:], ytr)
# train_loader_class = DataLoader(train_dataset_class, batch_size=32, shuffle=True)

# for input_dict in train_loader_class:
#     xw, xd  = input_dict["wide"].to(device), input_dict["deep"].to(device)
#     y = y.to(device)
#     y_pred = model(xw, xd) or y_pred = model(**input_dict) # **input_dict is used to unpack the dictionary into the model's input arguments
#     l = loss(y_pred, y)
#     total_loss += l.item()
#     l.backward()
#     optimizer.step()
#     optimizer.zero_grad()

class WideandDeepV3(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_features, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(40 + n_features, 1)

    def forward(self, x_wide, x_deep):
        deep_output = self.deep_stack(x_deep)
        wide_and_deep = torch.concat([x_wide, deep_output], dim=1)
        return self.output_layer(wide_and_deep)


# Multiple outputs
class WideandDeepV4(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.deep_stack = nn.Sequential(
            nn.Linear(n_features, 50),
            nn.ReLU(),
            nn.Linear(50, 40),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(40 + n_features, 1)
        self.aux_output_layer = nn.Linear(40, 1)

    def forward(self, xw, xd):
        deep_output = self.deep_stack(xd)
        wide_and_deep = torch.concat([xw, deep_output], dim=1)
        main_output =  self.output_layer(wide_and_deep)
        aux_output = self.aux_output_layer(deep_output)
        return main_output, aux_output
    




def train(model, optimizer, loss, train_loader, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in train_loader:
            x,y = x.to(device), y.to(device)
            y_pred = model(x)
            l = loss(y_pred, y)
            total_loss += l.item()
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

        mean_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {mean_loss:.4f}")


torch.manual_seed(42)
n_features = xtr.shape[1]
model = WideandDeep(n_features).to(device)
lr = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
mse = nn.MSELoss()
 

train(model, optimizer, mse, train_loader, 20)