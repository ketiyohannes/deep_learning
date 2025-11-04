import torch

x = torch.tensor([[1.0, 4.0, 7.0], [2.0, 3.0, 6.0]])
print(x)

print(x.shape)
print(x.dtype)

print(10 * (x + 1))
print(x.exp())

if torch.cuda.is_available():
    device = "cuda"
    print(device)
elif torch.backends.mps.is_available():
    device = "mps"
    print(device)
else:
    device = "cpu"
    print(device)


m = x.to(device)
print(m.device)


a = torch.tensor(5.0, requires_grad=True)
#forward pass
f = a ** 2
print(f)

#backward pass
f.backward()
print(a.grad)

learning_rate = 0.1
with torch.no_grad():
    a -= learning_rate * a.grad

print(a)

#also can use the .detach() method, y = x.detach()


a.grad.zero_()
