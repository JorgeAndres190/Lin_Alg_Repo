import torch

x = torch.tensor(1.0, requires_grad=True)
y = torch.tensor(-3.0, requires_grad=True)

f = (y*x**3 + 2*x**3 + x - 5*x) / (4*x**2*y**2 + 3)

f.backward()

print(x.grad)
print(y.grad)