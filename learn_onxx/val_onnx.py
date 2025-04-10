import onnxruntime
import torch
import numpy as np

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asinh(x) # asinh already defined in ATen

from torch.onnx.symbolic_registry import register_op

def asinh_symbolic(g, input, *, out=None):
    return g.op("Asinh", input) # onnx has Asinh op

register_op('asinh', asinh_symbolic, '', 9) # bind ATen::asinh to onnx.Asinh

model = Model()
input = torch.rand(1, 3, 10, 10)
torch.onnx.export(model, input, 'asinh.onnx')

torch_output = model(input).detach().numpy()

sess = onnxruntime.InferenceSession('asinh.onnx', providers=['CPUExecutionProvider'])
ort_output = sess.run(None, {'0': input.numpy()})[0]
print(np.allclose(torch_output, ort_output))
