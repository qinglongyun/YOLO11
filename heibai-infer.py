import os
import numpy as np
import cv2  # 确保你已经安装了 opencv-python: pip install opencv-python
from ultralytics import YOLO

# --- 步骤 1: 您的原始预测代码 (基本保持不变) ---

# 加载您训练好的模型
model = YOLO(r"C:\Users\model\ultralytics-yolo11-main.mg12.5\runs\train\segn-100-copy0.5-1280\weights\best.pt")

# 对指定的图像文件夹进行推理，并设置各种参数
results = model.predict(
    source="data/yuce", # 数据来源
    project='runs/detect',
    name='exp',
    conf=0.45,
    iou=0.35,
    imgsz=1280,
    half=False,
    device=None,
    max_det=300,
    agnostic_nms=True,# 保持True，得到无边框的掩码,保证无直线边界高质量掩码
    retina_masks=True,# 保持True，得到带边界的掩码
    save=True,  # 保持True，这样您会同时得到彩色的结果
    save_txt=True, # 保持True，得到边界框的txt文件
    show_boxes=False,
    line_width=1
    # 其他您不关心的参数为了简洁已省略
)

# --- 步骤 2: 新增的后处理代码，用于生成和保存黑白掩码 ---

print("\n预测完成。现在开始生成带边界的黑白二值掩码...")

# 定义输出目录
mask_output_dir = os.path.join(model.predictor.save_dir, 'binary_masks_with_borders')
os.makedirs(mask_output_dir, exist_ok=True)

# 定义腐蚀操作的核心
kernel = np.ones((3, 3), np.uint8)  # 3x3的腐蚀核心，可以腐蚀掉1个像素的边缘

for result in results:
    if result.masks is None:
        continue

    # 创建一个与原图一样大的黑色画布
    h, w = result.orig_shape
    bordered_mask_image = np.zeros((h, w), dtype=np.uint8)

    # 遍历每一个独立的掩码
    for mask_tensor in result.masks.data:
        # 将单个掩码转为NumPy数组
        single_mask = mask_tensor.cpu().numpy().astype(np.uint8) * 255

        # 对这个掩码进行腐蚀操作
        eroded_mask = cv2.erode(single_mask, kernel, iterations=2)

        # 将腐蚀后的掩码“画”到我们的黑色画布上
        bordered_mask_image[eroded_mask > 0] = 255

    # 构建保存路径
    original_filename = os.path.basename(result.path)
    base_name, _ = os.path.splitext(original_filename)
    mask_filename = f"{base_name}_mask_bordered.png"
    save_path = os.path.join(mask_output_dir, mask_filename)

    # 保存最终结果
    cv2.imwrite(save_path, bordered_mask_image)

print(f"带边界的黑白掩码已成功保存到目录: {mask_output_dir}")