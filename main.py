from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim

data_path = './'

cifar10 =  datasets.CIFAR10(data_path, train=True, download=True)
cifar10_val = datasets.CIFAR10(data_path, train=False, download=True)

type(cifar10).__mro__ # method resolution order
len(cifar10)

img, label = cifar10[99]
img, label

# plt.imshow(img)
# plt.show()
#

dir(transforms)

to_tensor = transforms.ToTensor()
img_t = to_tensor(img)
img_t.shape

tensor_cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.ToTensor())
img_t,_ = tensor_cifar10[99]

type(img_t)

imgs = torch.stack([img_t for img_t ,_ in tensor_cifar10],dim=3)
imgs.shape
imgs.view(3,-1).mean(dim=1)
imgs.view(3, -1).std(dim=1)
transforms.Normalize((0.4915, 0.4823, .4468), (0.2470, 0.2435, 0.2616))
transformed_cifar10 = datasets.CIFAR10(data_path, train=True, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468),
                         (0.2470, 0.2435, 0.2616))
]))

transformed_cifar10_val = datasets.CIFAR10(data_path, train=False, download=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4915, 0.4823, 0.4468),
                         (0.2470, 0.2435, 0.2616))
]))

img_t , _ = transformed_cifar10[99]

plt.imshow(img_t.permute(1,2,0))

plt.show()


label_map = {0:0, 2:1}
class_names = ['airplane', 'bird']

cifar2 = [(img, label_map[label])
for img, label in transformed_cifar10
if label in [0, 2]]

cifar2_val = [(img, label_map[label])
for img, label in transformed_cifar10_val
if label in [0, 2]]

n_out = 2
model = nn.Sequential(
    nn.Linear(
        3072, 512
    ),
    nn.Tanh(),
    nn.Linear(512, n_out)
)

def softmax(x):
    return torch.exp(x) / torch.exp(x).sum()

x = torch.tensor([1.0,2.0, 3.0])

softmax(x)
softmax(x).sum()

softmax = nn.Softmax(dim=1)

x = torch.tensor([[1.0,2.0,3.0],
                  [1.0,2.0,3.0]])


model = nn.Sequential(
        nn.Linear(3072, 512),
        nn.Tanh(),
        nn.Linear(512, 2),
        nn.Softmax(dim=1)
    )


img, _ = cifar2[99]

plt.imshow(img.permute(1,2,0))

try:
    img_batch = img.view(-1).unsqueeze(0)
except Exception as e:
    print(e)

out = model(img_batch)
out

_, index = torch.max(out, dim=1)

index


out = torch.tensor([
    [0.6, 0.4],
    [0.9,0.1],
    [0.3, 0.7],
    [0.2, 0.8]
])

class_index = torch.tensor([0,0,1,1]).unsqueeze(1)

truth = torch.zeros((4,2))
truth.scatter_(dim=1, index=class_index, value=1.0)
truth


model = nn.Sequential(
            nn.Linear(3072, 512),
            nn.Tanh(),
            nn.Linear(512, 2),
            nn.LogSoftmax(dim=1)
)

learning_rate = 1e-2

optimizer = optim.SGD(model.parameters(),lr=learning_rate)

loss_fn = nn.NLLLoss()
n_epochs = 100

# for epoch in range(n_epochs):
#     for img, label in cifar2:
#         out = model(img.view(-1).unsqueeze(0))
#         loss = loss_fn(out, torch.tensor([label]))

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print("Epoch: %d, Loss %f" % (epoch, float(loss)))

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

train_loader = torch.utils.data.DataLoader(cifar2, batch_size=64, shuffle=True)

model = nn.sequential(
    nn.Linear(3072, 512),
    nn.Tanh(),
    nn.Linear(512, 2),
    nn.LogSoftmax(dim=1)
)
