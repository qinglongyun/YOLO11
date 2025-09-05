import os
import numpy as np
import cv2
import csv
from ultralytics import YOLO

# ================================================================= #
# == 步骤 1: 关键参数设置                                          == #
# ================================================================= #
PIXELS_PER_MM = 20
MM_PER_PIXEL = 1.0 / PIXELS_PER_MM

# --- 您的预测代码 (保持不变) ---
model = YOLO(r"C:\Users\model\ultralytics-yolo11-main.mg12.5\runs\train\segn-100-copy0.5-1280\weights\best.pt")

results = model.predict(
    source="data/yuce",
    project='runs/detect',
    name='exp_with_FERET_measurements',  # 使用新的名字保存最终结果
    conf=0.45,
    iou=0.35,
    imgsz=1280,
    max_det=300,
    agnostic_nms=True,# 保持True，得到无边框的掩码,保证无直线边界高质量掩码
    retina_masks=True,# 保持True，得到带边界的掩码
    save=False,
    save_txt=True,
    show_boxes=False,
    line_width=1
)

# --- 步骤 2: 使用费雷特直径 (最小外接矩形) 进行最终测量 ---
print("\n预测完成。现在使用费雷特直径进行最终测量...")

output_dir = model.predictor.save_dir
os.makedirs(output_dir, exist_ok=True)

all_stones_data = []

for i, result in enumerate(results):
    if result.masks is None:
        print(f"在图像 {os.path.basename(result.path)} 中未检测到任何物体。")
        continue

    annotated_image = result.plot(boxes=False, line_width=2)

    # 遍历这张图片中的每一个高精度轮廓
    for j, contour in enumerate(result.masks.xy):
        if len(contour) >= 5:
            contour_cv = contour.astype(np.int32)

            # ================================================================= #
            # == 关键修复：使用最小外接矩形 (MinAreaRect) 代替椭圆拟合        == #
            # ================================================================= #
            # rect 是一个元组: ((center_x, center_y), (width, height), angle)
            rect = cv2.minAreaRect(contour_cv)

            # 获取矩形的四个顶点，用于绘制
            box = cv2.boxPoints(rect)
            box = np.int0(box)  # 将顶点坐标转换为整数

            # 矩形的尺寸 (width, height)
            width, height = rect[1]

            # 费雷特直径
            feret_major_px = max(width, height)  # 长边 -> 最大费雷特直径
            feret_minor_px = min(width, height)  # 短边 -> 最小费雷特直径

            # 转换为毫米
            feret_major_mm = feret_major_px * MM_PER_PIXEL
            feret_minor_mm = feret_minor_px * MM_PER_PIXEL

            # --- 在图像上进行可视化 ---
            # 1. 绘制最小外接矩形
            cv2.drawContours(annotated_image, [box], 0, (0, 0, 255), 2)  # 用红色绘制矩形

            # 2. 准备并标注文本
            text = f"D: {feret_major_mm:.2f}mm"
            center_point = (int(rect[0][0]), int(rect[0][1]))
            text_pos = (center_point[0] - 40, center_point[1] + 10)
            cv2.putText(annotated_image, text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            # --- 收集数据 ---
            all_stones_data.append({
                'image_name': os.path.basename(result.path),
                'stone_id': j + 1,
                'feret_major_mm': feret_major_mm,
                'feret_minor_mm': feret_minor_mm,
                'center_x_px': rect[0][0],
                'center_y_px': rect[0][1]
            })

    save_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(result.path))[0]}_measured.png")
    cv2.imwrite(save_path, annotated_image)

# --- 步骤 3: 保存CSV文件 (更新了列名) ---
if all_stones_data:
    csv_path = os.path.join(output_dir, 'particle_size_FERET_measurements.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image_name', 'stone_id', 'feret_major_mm', 'feret_minor_mm', 'center_x_px', 'center_y_px']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stones_data)
    print(f"\n所有费雷特直径测量数据已成功保存到: {csv_path}")