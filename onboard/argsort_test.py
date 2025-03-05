import torch
import torch.onnx

# 自定义实现 argsort
def custom_argsort(x, dim):
    values, indices = torch.sort(x, dim=dim)
    return indices

def replace_argsort_with_custom(model):
    # 将模型转换为 FX 图
    graph_module = torch.fx.symbolic_trace(model)
    for node in graph_module.graph.nodes:
        if node.op == 'call_function' and node.target == torch.argsort:
            # 获取 argsort 的参数
            args = list(node.args)
            kwargs = node.kwargs
            # 创建一个新的节点，调用 custom_argsort
            with graph_module.graph.inserting_after(node):
                new_node = graph_module.graph.call_function(custom_argsort, args=args, kwargs=kwargs)
            # 替换原节点的输出
            node.replace_all_uses_with(new_node)
            # 删除原节点
            graph_module.graph.erase_node(node)
    # 重新编译图
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module


# 定义一个简单的模型
class Model(torch.nn.Module):
    def forward(self, x):
        return torch.argsort(x, dim=1)
        # return custom_argsort(x, dim=1) # good

model = Model()

# modified_model = replace_argsort_with_custom(model)

input_tensor = torch.randn(1, 10)


from torch.onnx import register_custom_op_symbolic
# register_custom_op_symbolic("aten::argsort", custom_argsort, opset_version=13)
torch.argsort = custom_argsort
# 导出模型到 ONNX，使用更高的算子集版本（如 15）
torch.onnx.export(
    model,
    input_tensor,
    "model.onnx",
    opset_version=13,  # 提高算子集版本
    input_names=['input'],
    output_names=['output']
)
