import openslide
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models

class WSIGradCAM:
    def __init__(self, model, target_layer, device="cuda"):
        self.model = model.to(device)
        self.device = device
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

    def generate_for_patch(self, patch_tensor, class_idx=None):
        """为单个patch生成Grad-CAM"""
        patch_tensor = patch_tensor.to(self.device)
        patch_tensor.requires_grad_(True)

        # 前向传播
        output = self.model(patch_tensor.unsqueeze(0))

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 计算权重
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # 加权激活图
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = torch.relu(cam)

        # 归一化
        cam -= torch.min(cam)
        cam /= torch.max(cam)

        return cam.squeeze().cpu().detach().numpy()

    def process_wsi(self, wsi_path, patch_size=256, level=0, stride=None):
        """处理整个WSI"""
        if stride is None:
            stride = patch_size

        # 打开WSI
        slide = openslide.OpenSlide(wsi_path)

        # 获取WSI尺寸
        w, h = slide.level_dimensions[level]

        # 计算patch数量
        n_w = (w - patch_size) // stride + 1
        n_h = (h - patch_size) // stride + 1

        # 初始化热力图
        heatmap = np.zeros((h, w), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)

        # 预处理变换
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # 遍历所有patch
        for i in range(4):
            for j in range(4):
                x = i * stride
                y = j * stride

                # 读取patch
                patch = slide.read_region((x, y), level, (patch_size, patch_size))
                patch = patch.convert("RGB")

                # 预处理
                patch_tensor = transform(patch)

                try:
                    # 生成CAM
                    cam = self.generate_for_patch(patch_tensor)

                    # 将结果放入热力图
                    heatmap[y : y + patch_size, x : x + patch_size] += cv2.resize(
                        cam, (patch_size, patch_size)
                    )
                    count[y : y + patch_size, x : x + patch_size] += 1
                except Exception as e:
                    print(f"Error processing patch at ({x}, {y}): {e}")
                    continue

        # 平均重叠区域
        heatmap = np.divide(heatmap, count, where=count != 0)

        # 关闭WSI
        slide.close()

        return heatmap


def visualize_wsi_cam(wsi_path, heatmap, alpha=0.5, level=0):
    """可视化WSI和热力图"""
    # 读取WSI缩略图
    slide = openslide.OpenSlide(wsi_path)
    thumb = slide.get_thumbnail(slide.level_dimensions[level])
    slide.close()

    # 调整热力图大小
    heatmap_resized = cv2.resize(heatmap, thumb.size)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    heatmap_pil = Image.fromarray(heatmap_color)

    # 叠加显示
    blended = Image.blend(thumb.convert("RGB"), heatmap_pil, alpha)

    # 绘制
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(thumb)
    plt.title("Original WSI")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_pil)
    plt.title("Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(blended)
    plt.title("Overlay")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 1. 加载模型
    print("Loading model...")
    # model = torch.hub.load("pytorch/vision", "resnet50", pretrained=True)
    model = models.resnet50(pretrained=True)
    model.eval()
    print("Model loaded.")

    # 2. 选择目标层
    target_layer = model.layer4[-1].conv3

    # 3. 初始化WSI Grad-CAM
    wsi_gradcam = WSIGradCAM(model, target_layer)

    # 4. 处理WSI (替换为你的WSI路径)
    wsi_path = "hotmap/wsi.svs"
    heatmap = wsi_gradcam.process_wsi(wsi_path, patch_size=512, level=0)

    # 5. 可视化结果
    visualize_wsi_cam(wsi_path, heatmap)
