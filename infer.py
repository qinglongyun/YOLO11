from ultralytics import YOLO

# 加载预训练的YOLOv11n模型
model = YOLO(r"C:\Users\model\ultralytics-yolo11-main.mg12.5\runs\train\segn-100-copy0.5-1280\weights\best.pt")

# 对指定的图像文件夹进行推理，并设置各种参数
results = model.predict(
    source="data/yuce", # 数据来源，可以是文件夹、图片路径、视频、URL，或设备ID（如摄像头）
    project='runs/detect',
    name='exp',
    conf=0.54,  # 置信度阈值
    iou=0.35,  # IoU 阈值
    imgsz=1280,  # 图像大小
    half=False,  # 使用半精度推理
    device=None,  # 使用设备，None 表示自动选择，比如'cpu','0'
    max_det=300,  # 最大检测数量
    vid_stride=1,  # 视频帧跳跃设置
    stream_buffer=False,  # 视频流缓冲
    visualize=False,  # 可视化模型特征
    augment=False,  # 启用推理时增强
    agnostic_nms=True,  # 启用类无关的NMS
    classes=None,  # 指定要检测的类别
    retina_masks=True,  # 使用高分辨率分割掩码
    embed=None,  # 提取特征向量层
    show=False,  # 是否显示推理图像
    save=True,  # 保存推理结果
    save_frames=False,  # 保存视频的帧作为图像
    save_txt=True,  # 保存检测结果到文本文件
    save_conf=False,  # 保存置信度到文本文件
    save_crop=False,  # 保存裁剪的检测对象图像
    show_labels=True,  # 显示检测的标签
    show_conf=False,  # 显示检测置信度
    show_boxes=True,  # 显示检测框
    line_width=1  # 设置边界框的线条宽度，比如2，4
)