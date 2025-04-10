import onnx
from onnx import helper
from onnx import TensorProto
import onnxruntime
import numpy as np

a = helper.make_tensor_value_info('a', TensorProto.FLOAT, [10, 10])
x = helper.make_tensor_value_info('x', TensorProto.FLOAT, [10, 10])
b = helper.make_tensor_value_info('b', TensorProto.FLOAT, [10, 10])
output = helper.make_tensor_value_info('output', TensorProto.FLOAT, [10, 10])

mul = helper.make_node('Mul', ['a', 'x'], ['c'])
add = helper.make_node('Add', ['c', 'b'], ['output'])

graph = helper.make_graph([mul, add], 'linear_func', [a, x, b], [output])

model = helper.make_model(graph)
onnx.checker.check_model(model)
print(model)
onnx.save(model, 'linear_func.onnx')


# sess = onnxruntime.InferenceSession('linear_func.onnx', providers=['CPUExecutionProvider'])
# a = np.random.rand(10, 10).astype(np.float32)
# b = np.random.rand(10, 10).astype(np.float32)
# x = np.random.rand(10, 10).astype(np.float32)
# output = sess.run(['output'], {'a': a, 'b': b, 'x': x})[0]
# print(np.allclose(output, a * x + b))

model = onnx.load('linear_func.onnx')
node = model.graph.node
node[1].op_type = 'Sub'
onnx.checker.check_model(model)
onnx.save(model, 'linear_func_2.onnx')