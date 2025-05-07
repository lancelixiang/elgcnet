import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class MLPWithGradCAM(nn.Module):
    def __init__(self, input_dim):
        super(MLPWithGradCAM, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output = nn.Linear(32, 2)

        # 用于Grad-CAM的钩子
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)

        # 第一个隐藏层作为目标层
        target_activations = x

        if self.training:
            x.register_hook(self.activations_hook)
        self.activations = target_activations

        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        return self.activations


class MLPGradCAM:
    def __init__(self, model, target_layer_name="layer1"):
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

    def generate_cam(self, input_tensor, class_idx=None):
        input_tensor.requires_grad_(True)

        # 前向传播
        output = self.model(input_tensor.unsqueeze(0))

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 获取梯度和激活
        gradients = self.model.get_activations_gradient()
        activations = self.model.get_activations(input_tensor)

        # 全局平均池化梯度
        weights = torch.mean(gradients, dim=1, keepdim=True)

        # 加权激活
        cam = torch.sum(weights * activations, dim=1)
        cam = torch.relu(cam)

        # 归一化
        # cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.detach().cpu().numpy()


# 准备数据
data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y = data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 转换为PyTorch张量
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# 训练模型
model = MLPWithGradCAM(input_dim=X.shape[1])
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

# 创建Grad-CAM实例
mlp_gradcam = MLPGradCAM(model)

# 选择一个测试样本
sample_idx = 0
sample = X_test[sample_idx]
true_label = y_test[sample_idx].item()

# 生成CAM
cam = mlp_gradcam.generate_cam(sample)

print('xxxxxxxxxx cam', cam)

# 可视化
plt.figure(figsize=(15, 5))

# 原始特征重要性
plt.subplot(1, 2, 1)
plt.bar(range(len(data.feature_names)), sample.detach().numpy())
plt.xticks(range(len(data.feature_names)), data.feature_names, rotation=90)
plt.title("Original Feature Values")
plt.ylabel("Value")

# Grad-CAM特征重要性
plt.subplot(1, 2, 2)
plt.bar(range(len(data.feature_names)), cam)
plt.xticks(range(len(data.feature_names)), data.feature_names, rotation=90)
plt.title("Grad-CAM Feature Importance")
plt.ylabel("Importance Score")

plt.suptitle(f"True Label: {data.target_names[true_label]}")
plt.tight_layout()
plt.show()
