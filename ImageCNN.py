import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

!pip install Kaggle

from google.colab import files
uploaded = files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d samuelcortinhas/apples-or-tomatoes-image-classification

!unzip apples-or-tomatoes-image-classification.zip



!ls

# 데이터 전처리 및 데이터로더 설정
batch_size = 4
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.ImageFolder(root='./train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='./test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('apples', 'tomatoes')

import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64 * 56 * 56, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 2))

        torch.nn.init.xavier_uniform_(self.fc[0].weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out



cnn_net = CNN()
cnn_net = CNN().to(device)

input_shape = (3, 224, 224)  # 예제 입력 형태 (RGB 이미지, 높이 224, 너비 224)
inputs = torch.randn(1, *input_shape).to(device)  # 입력 데이터 생성 및 디바이스로 이동
summary(cnn_net, input_size=input_shape)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn_net.parameters(), lr=0.001, momentum=0.9)

loss_history = []
accuracy_history = []



# 훈련 루프
for epoch in range(50):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = cnn_net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if i % 2000 == 1999:
            loss_history.append(running_loss / 2000)
            accuracy = 100 * correct / total
            accuracy_history.append(accuracy)

            print('[%d, %5d] loss: %.3f accuracy: %.2f %%' % (epoch + 1, i + 1, running_loss / 2000, accuracy))
            running_loss = 0.0

print('Finished Training')

# 손실 및 정확도 그래프 출력
plt.figure(figsize=(12, 5))

# Loss 그래프
plt.subplot(1, 2, 1)
plt.plot(loss_history, color='blue', linestyle='-', label='Loss')
plt.title('Training Loss')
plt.xlabel('Mini-Batch Iterations')
plt.ylabel('Loss')

# Accuracy 그래프
plt.subplot(1, 2, 2)
plt.plot(accuracy_history, color='green', linestyle='-', label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Mini-Batch Iterations')
plt.ylabel('Accuracy (%)')


plt.show()

PATH = './cnn_net.pth'
torch.save(cnn_net.state_dict(), PATH)


# 테스트 이미지 분류
dataiter = iter(testloader)
images, labels = next(dataiter)

# 이미지 확인하기
def imshow(img):
    img = img / 2 + 0.5     # 정규화 해제
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# 학습용 이미지 뽑기
dataiter = iter(trainloader)
images, labels = next(dataiter)  # .next() 대신 .next()를 사용합니다.

# 이미지 보여주기
imshow(torchvision.utils.make_grid(images))

# 이미지별 라벨 (클래스) 보여주기
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))




cnn_net = CNN()
cnn_net.load_state_dict(torch.load(PATH))
cnn_net.to(device)

outputs = cnn_net(images.to(device))
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%4s' % classes[predicted[j]] for j in range(4)))


# 최종 모델 평가
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = cnn_net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %.2f %%' % (100 * correct / total))

class_correct = np.zeros(2)
class_total = np.zeros(2)

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)  # labels를 GPU로 이동
        outputs = cnn_net(images.to(device))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        label = labels.cpu().numpy()
        class_correct[label] += c.cpu().numpy()
        class_total[label] += 1

for i in range(2):
    accuracy = 100 * class_correct[i] / class_total[i]
    print('Accuracy of %5s : %2d %%' % (classes[i], accuracy))