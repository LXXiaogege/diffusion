import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import BasicUNet
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
print("Using device :", device)
dataset = datasets.MNIST(root='/Users/lvxin/datasets/mnsit', download=True, train=True,
                         transform=transforms.ToTensor())

data_loader = DataLoader(dataset, batch_size=128, shuffle=True, drop_last=True)
model = BasicUNet().to(device)

loss_func = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=0.001)

epoch = 1

losses = []
for i in range(epoch):
    for x, y in tqdm(data_loader, total=len(data_loader), desc="step in every epoch",position=0):
        pred = model(x)
        loss = loss_func(pred, x)  # 预测图像与原图的loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    avg_loss = sum(losses[-len(data_loader):]) / len(data_loader)
    print(f"epoch {epoch} avg loss :", avg_loss)