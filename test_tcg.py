"""
Some simple tests for TCG optimizer
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tcg import TCG

seed = 1
cuda = torch.cuda.is_available()
torch.manual_seed(seed)
device = torch.device("cuda" if cuda else "cpu")

x_dim, y_dim = 1, 1

# One linear layer network
# f(x, bias, weight) = bias + weight*x
class Lin(nn.Module):
    def __init__(self):
        super(Lin, self).__init__()
        self.layer = nn.Linear(x_dim, y_dim)
        self.layer.bias.data.fill_(0.)
        self.layer.weight.data.fill_(0.)
    def forward(self, x):
        y = self.layer(x)
        return y

# MSE loss
# criterion(y, yt) = ||y-yt||**2 / y_dim
def criterion(y, yt):
    return torch.mean((y - yt)**2)

model = Lin()

# Call the optimizer with parameters
optimizer = TCG(model.parameters(),
                delta=2,
                kmax=10,
                tol=10**-6,
                eps=10**-6,
                momentum=0.)

# test 1:
# x = (0, 1), yt = 0 + 1*x
bs = 2
x = torch.zeros(bs, x_dim)
x[1, 0] = 1.
yt = torch.zeros(bs, y_dim)
yt[1, 0] = 1.

optimizer.zero_grad()
y = model(x)
loss = criterion(y, yt)
# no call to loss.backward()
flag, k, val = optimizer.step(loss) # loss should be given to step
with torch.no_grad():
     print("loss before: %.4f"%(loss.item()))
     y = model(x)
     loss = criterion(y, yt)
     print("loss after: %.4f"%(loss.item()))
     print("flag, k, val= %d, %d, %.4f"%(flag, k, val))
     print("bias= "+str(model.layer.bias.data))
     print("weight= "+str(model.layer.weight.data))

# test 2:
# x random, yt = 1 + 2*x
bs = 10
x = torch.randn(bs, x_dim)
yt = 1 + 2*x

optimizer.zero_grad()
y = model(x)
loss = criterion(y, yt)
# no call to loss.backward()
flag, k, val = optimizer.step(loss) # loss should be given to step
with torch.no_grad():
     print("loss before: %.4f"%(loss.item()))
     y = model(x)
     loss = criterion(y, yt)
     print("loss after: %.4f"%(loss.item()))
     print("flag, k, val= %d, %d, %.4f"%(flag, k, val))
     print("bias= "+str(model.layer.bias.data))
     print("weight= "+str(model.layer.weight.data))

