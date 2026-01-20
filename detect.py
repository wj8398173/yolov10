from ultralytics import YOLOv10
import json
# model = YOLOv10.from_pretrained ('jameslahm/yolov10{n/s/m/b/l/x}')
# or
# wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10{n/s/m/b/l/x}.pt
model = YOLOv10(r'E:\yolov10\runs\detect\train\weights\best.pt')
img_url=r'E:\yolov10\ultralytics\assets'

results=model.predict(img_url,save=True)


#.................................优化后的推理代码.............................................
# from ultralytics import YOLOv10
# import torch
# import os
# from pathlib import Path
#
#
# class YOLOv10InferenceOptimizer:
#     def __init__(self):
#         # 硬件配置
#         self.device = '0' if torch.cuda.is_available() else 'cpu'
#         self.half_precision = True if self.device != 'cpu' else False
#
#         # 路径配置（适配Windows路径）
#         self.model_path = r'E:\yolov10\runs\detect\train5\weights\best.pt'
#         self.source = [
#             r'E:\yolov10\ultralytics\assets',  # 图片目录
#             # r'E:\test_videos\demo.mp4',      # 视频文件
#             # '0'                              # 摄像头
#         ]
#
#         # 输出配置
#         self.project = 'runs/detect'
#         self.name = 'prod_exp'
#         self.save_dir = Path(self.project) / self.name
#
#     def setup_model(self):
#         """初始化并优化模型加载"""
#         print(f"Loading model from {self.model_path}")
#
#         # 自动识别输入尺寸
#         model = YOLOv10(self.model_path)
#         self.imgsz = model.args['imgsz'] if 'imgsz' in model.args else 640
#
#         # 优化模型配置
#         return model.to(self.device).half() if self.half_precision else model.to(self.device)
#
#     def configure_predict_args(self):
#         """配置预测参数"""
#         return {
#             'source': self.source,
#             'conf': 0.4,  # 置信度阈值
#             'iou': 0.45,  # NMS IoU阈值
#             'imgsz': self.imgsz,
#             'device': self.device,
#             'stream': False,  # 非实时处理时关闭
#             'max_det': 300,  # 最大检测目标数
#             'agnostic_nms': False,  # 类别无关NMS
#             'augment': True,  # 推理时增强(TTA)
#             'visualize': False,  # 特征可视化
#             'save': True,
#             'save_txt': True,  # 保存标签文件
#             'save_conf': True,  # 保存置信度
#             'save_crop': False,  # 保存裁剪目标
#             'show_labels': True,
#             'show_conf': True,
#             'show_boxes': True,
#             'vid_stride': 2,  # 视频帧采样间隔
#             'line_width': None,  # 自动调整线宽
#             'half': self.half_precision,
#             'project': self.project,
#             'name': self.name,
#             'exist_ok': True
#         }
#
#     def run_inference(self, model):
#         """执行推理并返回结果"""
#         with torch.no_grad():
#             return model.predict(**self.configure_predict_args())
#
#     def post_process(self, results):
#         """后处理与性能分析"""
#         if isinstance(results, list):
#             # 打印FPS统计
#             total_time = sum(r.speed['inference'] for r in results)
#             avg_fps = len(results) / total_time
#             print(f"\nAverage FPS: {avg_fps:.1f} | Total processed: {len(results)} items")
#
#             # 保存统计结果
#             stats = {
#                 'total_images': len(results),
#                 'average_fps': avg_fps,
#                 'detection_counts': sum(len(r.boxes) for r in results)
#             }
#             with open(self.save_dir / 'stats.json', 'w') as f:
#                 json.dump(stats, f)
#
#         # 导出TensorRT引擎（可选）
#         model.export(format='engine', imgsz=self.imgsz, device=self.device)
#         print(f"Optimized model saved to {self.save_dir}")
#
#
# if __name__ == '__main__':
#     detector = YOLOv10InferenceOptimizer()
#
#     try:
#         # 初始化模型
#         model = detector.setup_model()
#
#         # 执行推理
#         results = detector.run_inference(model)
#
#         # 后处理
#         detector.post_process(results)
#
#     except Exception as e:
#         print(f"Error occurred: {str(e)}")
#         # 发送错误通知（可选）
#         # send_alert(f"Inference failed: {str(e)}")