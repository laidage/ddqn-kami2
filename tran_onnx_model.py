"""
将训练好的模型转化为onnx格式，便于使用，提高运行效率，
以及减小之后需要使用模型时需要导入的库（pytorch实在是太大了）
"""

import torch
import torch.nn
from net import Net

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_dtype(torch.float16)
kami_model = Net().to(device).half()

kami_model.load_state_dict(torch.load("model/net.pth"))
batch_size = 128
input_names = ['input']
output_names = ['output']
kami_model.eval()

x = torch.randn(batch_size, 5, 20, 15, requires_grad=True)
torch.onnx.export(kami_model, x, "model/kami.onnx", opset_version=11, dynamic_axes={'input':{0:'batch'}}, input_names=input_names, output_names=output_names,
 export_params=True, training=torch.onnx.TrainingMode.EVAL)