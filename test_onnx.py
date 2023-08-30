import os.path as osp
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

test_arr = np.random.randn(1, 3, 384, 640).astype(np.float32)

# model = torch.load(r'F:\Artificial_neural_Network\yolov8-main\weights\yolov8n.pt').cuda().eval()

# model = torch.load(r'F:\Artificial_neural_Network\yolov8-main\weights\yolov8n.pt')
#
# print('pytorch result:', model(torch.from_numpy(test_arr).cuda()))

model_onnx = onnx.load(r'E:\TensorRT-8.4.3.1\bin\yolov8n_384_640.onnx')
onnx.checker.check_model(model_onnx)

ort_session = ort.InferenceSession(r'E:\TensorRT-8.4.3.1\bin\yolov8n_384_640.onnx')
outputs = ort_session.run(None, {'images': test_arr})
print('onnx_result:', outputs)
print('onnx_result shape:', outputs.size())