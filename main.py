import torch
import torch.nn as nn
import torch.onnx as onnx
import onnxruntime as ort
import numpy as np

#导出、推理
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_relu_flatten_fc_stack = nn.Sequential(
            nn.conv2d(1, 10, kernel_size=5),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10 * 24 * 24, 10)
        )

    def forward(self, x):
        x = self.conv_relu_flatten_fc_stack(x)
        return x

model = MyModel()

input_data = torch.randn(1, 1, 28, 28)

onnx.export(
    mode,
    input_data,
    'MyModel.onnx',
    input_name = ['input_0'],
    output_name = ['output_0'],
    opset_version = 17, #ONNX操作集版本
    dynamic_axes = {'input_0': {0: 'batch_size'} } #这行不懂
)
print("模型已成功导出！文件名是'MyModel.onnx")

print("-"*35)
#用ONNX加载模型并推理，创建推理会话

session = ort.InferenceSession("MyModel.onnx") #导入模型

input_data = np.random.rand(1, 1, 28, 28).astype(np.float32) #准备输入数据，ort通常需要Numpy数组

#获取输入和输出名称
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

#执行推理，需要提供输入、输出的名称
output = session.run([output_name], {input_name: input_data})[0]

print(output.shape)
print(output[0][:5])











