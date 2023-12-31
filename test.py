import torch
p = torch.tensor([1, 2, 3, 4, 5])
p1 = p
p[1,2] = 10

print(p)
print(p1)