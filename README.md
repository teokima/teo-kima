import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, classification_report

# FashionMNIST 데이터셋 로드
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# 클래스 레이블 매핑
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

# 이미지 및 클래스 레이블 시각화 함수
def show_images(images, labels, predicted_labels=None):
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i][0], cmap=plt.cm.binary)
        true_label = classes[labels[i]]
        if predicted_labels is not None:
            true_label += f'\nPredicted: {classes[predicted_labels[i]]}'
        plt.xlabel(true_label)
    plt.show()

# 모델 성능 평가 및 결과 분석
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    predicted_labels = []
    true_labels = []
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.numpy())
            true_labels.extend(labels.numpy())

    print(f'Accuracy of the network on the test images: {100 * correct / total:.2f}%')

    # 추가적인 메트릭 계산 및 분석
    print('Classification Report:')
    print(classification_report(true_labels, predicted_labels, target_names=classes))

    # 혼동 행렬 시각화
    cm = confusion_matrix(true_labels, predicted_labels)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

# 학습 데이터셋에서 무작위 이미지 가져오기
trainloader = torch.utils.data.DataLoader(trainset, batch_size=25, shuffle=True)
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 이미지 시각화
show_images(images.numpy(), labels.numpy())

# 간단한 CNN 모델 구축
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 5)  # 입력 채널 수: 1 (흑백 이미지), 출력 채널 수: 32, 커널 크기: 5x5
        self.pool = nn.MaxPool2d(2, 2)    # 풀링 레이어: 최대 풀링, 커널 크기: 2x2
        self.conv2 = nn.Conv2d(32, 64, 3) # 입력 채널 수: 32, 출력 채널 수: 64, 커널 크기: 3x3
        self.fc1 = nn.Linear(64 * 5 * 5, 128) # fully connected layer, 입력 뉴런 수: 64*5*5, 출력 뉴런 수: 128
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 출력 클래스 수: 10 (FashionMNIST 클래스 수)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 모델 생성 및 컴파일
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 모델 훈련
epochs = 5
train_loss_history = []
for epoch in range(epochs):  
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(trainloader)
    train_loss_history.append(train_loss)
    print('Epoch [%d/%d], Loss: %.4f' % (epoch+1, epochs, train_loss))

# 손실 그래프
plt.plot(train_loss_history, label='Training Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 모델 평가 및 결과 분석
evaluate_model(net, torch.utils.data.DataLoader(testset, batch_size=25, shuffle=False))
