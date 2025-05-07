import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class TransformerWithGradCAM(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2):
        super(TransformerWithGradCAM, self).__init__()
        self.embedding = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(64, num_classes)
        
        # 存储attention权重和梯度
        self.attention_weights = None
        self.attention_gradients = None
        
    def save_attention_weights(self, module, input, output):
        # 存储attention权重 [batch, heads, seq_len, seq_len]
        self.attention_weights = output[0]
        
    def save_attention_gradients(self, grad):
        self.attention_gradients = grad
        
    def forward(self, x):
        # 嵌入层
        x = self.embedding(x)  # [batch, seq_len, d_model]
        
        # 注册attention钩子
        handle = self.transformer.layers[0].self_attn.register_forward_hook(self.save_attention_weights)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 移除钩子
        handle.remove()
        
        # 分类
        x = x.mean(dim=1)  # 全局平均池化
        output = self.classifier(x)
        return output

class AttentionGradCAM:
    def __init__(self, model, target_layer='transformer'):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
     
    def generate_cam(self, input_tensor, class_idx=None):
        input_tensor.requires_grad = True
        
        # 前向传播
        output = self.model(input_tensor.unsqueeze(0))
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 获取attention权重和梯度
        weights = self.model.attention_weights  # [1, heads, seq_len, seq_len]
        gradients = self.model.attention_gradients  # [1, heads, seq_len, seq_len]
        
        if gradients is None:
            raise ValueError("No gradients captured. Check hook registration.")
        
        # 计算重要性权重
        alpha = gradients.unsqueeze(0).mean(dim=(2, 3), keepdim=True)  # [1, heads, 1, 1]
        
        # 加权attention
        cam = torch.sum(alpha * weights, dim=1)  # [1, seq_len, seq_len]
        cam = torch.relu(cam.squeeze(0))
        
        # 归一化
        cam -= cam.min()
        cam /= cam.max()
        
        return cam.detach().cpu().numpy()

# 示例数据（假设是时间序列数据）
seq_len = 20
input_dim = 10
X = torch.randn(5, seq_len, input_dim)  # 5个样本，每个20时间步，10特征

# 创建模型
model = TransformerWithGradCAM(input_dim=input_dim, num_classes=2)

# 注册梯度钩子
def backward_hook(module, grad_input, grad_output):
    model.attention_gradients = grad_output[0].detach()
    
handle = model.transformer.layers[0].self_attn.register_backward_hook(backward_hook)

# 训练模型（简化示例）
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模拟训练
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X)
    loss = criterion(outputs, torch.randint(0, 2, (5,)))
    loss.backward()
    optimizer.step()

# 创建Grad-CAM实例
attention_gradcam = AttentionGradCAM(model)

# 选择一个样本
sample = X[0]  # [seq_len, input_dim]

# 生成CAM
cam = attention_gradcam.generate_cam(sample)

# 可视化
plt.figure(figsize=(12, 8))

# Attention热力图
plt.subplot(2, 1, 1)
plt.imshow(cam, cmap='hot', interpolation='nearest')
plt.title("Attention Grad-CAM Heatmap")
plt.xlabel("Sequence Position")
plt.ylabel("Sequence Position")
plt.colorbar()

# 特征重要性
plt.subplot(2, 1, 2)
feature_importance = np.mean(cam, axis=1)  # 对时间步平均
plt.bar(range(seq_len), feature_importance)
plt.title("Feature Importance over Sequence")
plt.xlabel("Sequence Position")
plt.ylabel("Importance Score")

plt.tight_layout()
plt.show()

# 移除钩子
handle.remove()