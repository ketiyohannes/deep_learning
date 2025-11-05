import torch    
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms.v2 as T

device = 'cpu'
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print("Device: ", device)
# converting the data to tensors and scaling the data to the range [0, 1]
toTensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])

# loading the data
train_and_valid_data = torchvision.datasets.FashionMNIST(root="datasets", train=True, download=True,transform=toTensor)
test_data = torchvision.datasets.FashionMNIST(root="datasets", train=False, download=True, transform=toTensor)

torch.manual_seed(42)
train, valid = torch.utils.data.random_split(train_and_valid_data, [55000, 5000])

train_loader = DataLoader(train, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid, batch_size=32, shuffle=False)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)



class ImageClassifier(nn.Module):
    def __init__(self, n_inputs, hidden1, hidden2, output):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_inputs, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, output)
        )

    def forward(self, x):
        return self.model(x)


torch.manual_seed(42)
model = ImageClassifier(28*28, 300, 100, len([classes for classes in train_and_valid_data.classes])).to(device)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

def train(model, loss, optimizer, loader, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            x,y = x.to(device), y.to(device)
            y_pred = model(x)
            l = loss(y_pred, y)
            total_loss += l.item()
            l.backward()
            optimizer.step()
            optimizer.zero_grad()
        mean_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {mean_loss:.4f}")


train(model, loss, optimizer, train_loader, epochs=20)














