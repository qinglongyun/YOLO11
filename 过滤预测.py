import torch
import cv2
import os
from ultralytics import YOLO

# --- 步骤 0: 设置参数 ---
output_dir = 'runs/detect/exp_filtered_no_boxes'
min_area_threshold = 20000

os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)


# --- 步骤 1: 加载模型并进行预测 ---
model = YOLO(r"C:\Users\model\ultralytics-yolo11-main.mg12.5\runs\train\exp3\weights\best.pt")

results = model.predict(
    source="data/yuce",
    conf=0.45,
    iou=0.6,
    imgsz=640,
    agnostic_nms=True,
    retina_masks=True,
    save=False,
    save_txt=False,
)

# --- 步骤 2: 循环遍历结果并应用后处理过滤器 ---
print(f"开始后处理，过滤掉面积小于 {min_area_threshold} 像素的目标...")

for i, result in enumerate(results):
    if result.masks is None:
        print(f"在图像 {result.path} 中没有检测到任何目标，跳过。")
        continue

    original_masks = result.masks.data
    original_boxes = result.boxes.data

    areas = torch.sum(original_masks, dim=(1, 2))
    keep = areas >= min_area_threshold

    if not torch.any(keep):
        print(f"图像 {result.path} 中所有目标都小于阈值，不保存结果。")
        continue

    result.masks.data = original_masks[keep]
    result.boxes.data = original_boxes[keep]

    # --- 步骤 3: 手动保存过滤后的结果 ---
    # 1. 保存过滤后的可视化图像
    # ================================================================= #
    # == 关键修改在这里：使用正确的参数 labels=True == #
    # ================================================================= #
    img_processed = result.plot(boxes=False, line_width=1, labels=True)

    base_filename = os.path.basename(result.path)
    save_path_img = os.path.join(output_dir, base_filename)
    cv2.imwrite(save_path_img, img_processed)

    # 2. 保存过滤后的标签（.txt 文件）
    txt_filename = os.path.splitext(base_filename)[0] + '.txt'
    save_path_txt = os.path.join(output_dir, 'labels', txt_filename)

    filtered_boxes_xywhn = result.boxes.xywhn
    filtered_cls = result.boxes.cls

    with open(save_path_txt, 'w') as f:
        for i in range(len(filtered_boxes_xywhn)):
            cls_id = int(filtered_cls[i])
            x_center, y_center, width, height = filtered_boxes_xywhn[i]
            f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

print(f"处理完成！不带边框的过滤后结果已保存到 '{output_dir}' 目录。")