import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import requests
from io import BytesIO

# 定义 Grad-CAM 类
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        target_layer.register_forward_hook(self.save_activations)
        target_layer.register_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def forward(self, x):
        return self.model(x)
    
    def backward(self, output, class_idx):
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
    
    def generate(self, x, class_idx=None):
        # 确保输入需要梯度
        x.requires_grad_(True)
        
        # 前向传播
        output = self.forward(x)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # 反向传播
        self.backward(output, class_idx)
        
        # 计算权重
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        
        # 加权激活图
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        # 归一化
        cam -= torch.min(cam)
        cam /= torch.max(cam)
        
        return cam.squeeze().detach().cpu().numpy()

# 图像预处理
def preprocess_image(img_path, img_size=224):
    # 加载图像
    if img_path.startswith('http'):
        response = requests.get(img_path)
        img = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        img = Image.open(img_path).convert('RGB')
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)
    img_tensor.requires_grad_(True)  # 确保需要梯度
    return img_tensor, img

# 可视化函数
def visualize_cam(img, cam, alpha=0.5):
    # 调整热力图大小
    cam = cv2.resize(cam, (img.size[0], img.size[1]))
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    
    # 转换为 RGB
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = Image.fromarray(heatmap)
    
    # 叠加原始图像
    result = Image.blend(img, heatmap, alpha)
    
    # 显示
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap)
    plt.title('Heatmap')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(result)
    plt.title('Grad-CAM Result')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# 主函数
def main():
    # 加载预训练模型
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # 选择目标层 (通常是最后一个卷积层)
    target_layer = model.layer4[-1].conv3
    
    # 初始化 Grad-CAM
    grad_cam = GradCAM(model, target_layer)
    
    # 图像路径 (可以是本地文件或URL)
    img_path = "hotmap/both.png"
    
    # 预处理图像
    input_tensor, original_img = preprocess_image(img_path)
    
    # 生成 CAM
    cam = grad_cam.generate(input_tensor)
    
    # 可视化
    visualize_cam(original_img, cam)

if __name__ == "__main__":
    main()