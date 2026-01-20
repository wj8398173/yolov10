from ultralytics import YOLOv10


model_ymal_path=r'E:\yolov10\ultralytics\cfg\models\v10\yolov10s.yaml'

data_ymal_path=r'E:\yolov10\ultralytics\cfg\datasets\wjVOC.yaml'

pre_model_name=r'E:\yolov10\weights\yolov10s.pt'


if __name__ == '__main__':
    model=YOLOv10(model_ymal_path).load(pre_model_name)
    result=model.train(data=data_ymal_path,
                epochs=50,
                batch=2,
                imgsz=512)

#..........................优化后的训练代码......................................................
# from ultralytics import YOLOv10
# import torch
#
# # 配置路径
# model_yaml_path = r'E:\yolov10\ultralytics\cfg\models\v10\yolov10s.yaml'
# data_yaml_path = r'E:\yolov10\ultralytics\cfg\datasets\wjVOC.yaml'
# pretrained_weights = r'E:\yolov10\weights\yolov10s.pt'
#
#
# def main():
#     # 检查GPU可用性
#     device = '0' if torch.cuda.is_available() else 'cpu'
#     print(f"Using device: {torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'}")
#
#     # 初始化模型（优化初始化方式）
#     model = YOLOv10(model_yaml_path).load(
#         pretrained_weights,
#         task='detect',
#         imgsz=512,
#         device=device,
#         fuse=True  # 启用层融合加速
#     )
#
#     # 训练参数配置（优化版）
#     train_args = {
#         'data': data_yaml_path,
#         'epochs': 100,  # 增加训练轮次
#         'batch': 8,  # 根据GPU显存调整（RTX 3060 12G可用）
#         'imgsz': 512,
#         'device': device,
#         'workers': 4,  # 根据CPU核心数调整
#         'optimizer': 'AdamW',  # 使用更好的优化器
#         'lr0': 1e-4,  # 初始学习率
#         'lrf': 0.01,  # 最终学习率（lr0 * lrf）
#         'cos_lr': True,  # 余弦退火调度
#         'weight_decay': 0.05,
#         'label_smoothing': 0.1,
#         'augment': True,  # 启用数据增强
#         'mosaic': 0.5,  # Mosaic增强概率
#         'mixup': 0.2,  # Mixup增强概率
#         'close_mosaic': 10,  # 最后10epoch关闭mosaic
#         'amp': True,  # 自动混合精度训练
#         'resume': False,
#         'save_period': 10,  # 每10epoch保存检查点
#         'patience': 30,  # 早停机制
#         'box': 7.5,  # 调整box loss权重
#         'cls': 0.5,  # 调整分类loss权重
#         'dfl': 1.5,  # 调整dfl loss权重
#         'project': 'runs/train',
#         'name': 'yolov10s_custom',
#         'exist_ok': True
#     }
#
#     # 梯度累积配置（显存不足时使用）
#     if train_args['batch'] < 8:
#         train_args['accumulate'] = max(1, 8 // train_args['batch'])
#         print(f"Applying gradient accumulation: {train_args['accumulate']}")
#
#     # 启动训练
#     results = model.train(**train_args)
#
#     # 导出最佳模型（TensorRT格式）
#     model.export(format='engine', imgsz=[512, 512], device=0)
#
#
# if __name__ == '__main__':
#     main()



#终端命令行训练
# yolo detect train data=E:\yolov10\ultralytics\cfg\datasets\wjVOC.yaml model=E:\yolov10\ultralytics\cfg\models\v10\yolov10s.yaml epochs=50 batch=2 imgsz=512 device=0

#训练命令，带预训练权重
# yolo detect train data=E:\yolov10\ultralytics\cfg\datasets\wjVOC.yaml model=E:\yolov10\weights\yolov10s.pt  epochs=50 batch=2 imgsz=512 device=0
#训练命令，不带预训练权重
#yolo detect train data=E:\yolov10\ultralytics\cfg\datasets\wjVOC.yaml model=E:\yolov10\ultralytics\cfg\models\v10\yolov10s.yaml  epochs=50 batch=2 imgsz=512 device=0

# 只处理jpg文件
# yolo detect predict model=yolov10s.pt source="E:/test_images/*.jpg"
# 使用默认摄像头并实时显示
#yolo detect predict model=yolov10s.pt source=0 show=True
# 处理视频并保存结果
#yolo detect predict model=yolov10s.pt source="input.mp4" save=True

# 使用批量推理提升速度（需足够显存）
#yolo detect predict model=yolov10s.pt source="E:/test_images" batch=16


