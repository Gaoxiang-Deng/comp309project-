# %%
import torch
import torchvision

from torchvision import transforms

transform = transforms.Compose(
    [transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.Resize([32, 32]),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
transform_test = transforms.Compose(
    [transforms.RandomResizedCrop(224),
     transforms.RandomHorizontalFlip(),
     transforms.Resize([32, 32]),
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

classes = ('cherry', 'strawberry', 'tomato')

data_path = "C:/workspace/traindata/traindata"
data_all = torchvision.datasets.ImageFolder(root=data_path, transform=transform)
test_data_path = "C:/workspace/traindata/testdata"
test_data = torchvision.datasets.ImageFolder(root=test_data_path, transform=transform_test)

# %%
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from torch.utils.data import SubsetRandomSampler


def dataset_sampler(dataset, val_percentage=0.1):
    """
    split dataset into train set and val set
    :param dataset:
    :param val_percentage: validation percentage
    :return: split sampler
    """
    sample_num = len(dataset)
    file_idx = list(range(sample_num))
    train_idx, val_idx = train_test_split(file_idx, test_size=val_percentage, random_state=42)
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    return train_sampler, val_sampler


train_sampler, val_sampler = dataset_sampler(data_all)

batch_size = 20
trainloader = data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=train_sampler)
validloader = data.DataLoader(data_all, batch_size=batch_size, num_workers=0, sampler=val_sampler)

testloader = data.DataLoader(test_data, batch_size=batch_size, num_workers=0)
##获得一个batch的数据
for step, (b_x, b_y) in enumerate(trainloader):
    if step > 0:
        break

print(b_x.shape)
print(b_y.shape)
print("图像的取值范围为：", b_x.min(), "~", b_x.max())
print(classes)
# %%
import matplotlib.pyplot as plt
import numpy as np


def imshow(image, ax=None, title=None, normalize=True):
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


# %%
dataiter = iter(trainloader)
images, labels = dataiter.next()

fig, axes = plt.subplots(figsize=(12, 12), ncols=5)
print('training images')
for i in range(5):
    axe1 = axes[i]
    imshow(torchvision.utils.make_grid(images))

print(images[0].size())
print(','.join('%5s' % classes[labels[j]] for j in range(batch_size)))

# %%
num_epochs = 50
num_classes = 2
batch_size = 25
learning_rate = 0.001

epochs = 50


# %%
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(

            nn.Conv2d(3, 6, 5 , padding=2),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(1024*2*2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, xb):
        return self.network(xb)


import torch.optim as optim

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

# %%

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

# %%
train_losses = []
valid_losses = []
for epoch in range(1, num_epochs + 1):
    train_loss = 0.0
    valid_loss = 0.0
    net.train()
    for data, label in trainloader:
        data = data.to(device)
        target = label.to(device)
        optimizer.zero_grad()
        # forward-pass: compute-predicted-outputs-by-passing-inputs-to-the-model
        output = net(data)
        # calculate-the-batch-loss
        loss = criterion(output, target)
        # backward-pass: compute-gradient-of-the-loss-wrt-model-parameters
        loss.backward()
        # perform-a-ingle-optimization-step (parameter-update)
        optimizer.step()
        # update-training-loss
        train_loss += loss.item() * data.size(0)
        # validate-the-model
        net.eval()
    for data, target in validloader:
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        loss = criterion(output, target)
        # update-average-validation-loss
        valid_loss += loss.item() * data.size(0)

    # calculate-average-losses
    train_loss = train_loss / len(trainloader.sampler)
    valid_loss = valid_loss / len(validloader.sampler)
    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    # print-training/validation-statistics
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(epoch, train_loss, valid_loss))

print('Finish training')

# %%
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# %%
# test
test_dataiter = iter(testloader)

# print the image
image, labels = test_dataiter.next()
imshow(torchvision.utils.make_grid(image))
# %%
net = Net()
net.load_state_dict(torch.load(PATH))

# %%
outputs = net(image)

# %%
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

# %%
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

# %%
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)
        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print("Accuracy for class {:5s} is: {:.1f} %".format(classname, accuracy))

# %%


plt.plot(train_losses, label='Training loss')
plt.plot(valid_losses, label='Validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(frameon=False)

