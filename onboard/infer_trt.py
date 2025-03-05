import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import os
import torch

def load_serialized_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    try:
        with open(engine_file_path, "rb") as f:
            engine_data = f.read()
        engine = runtime.deserialize_cuda_engine(engine_data)
        if engine is None:
            raise RuntimeError("引擎反序列化失败")
        print(f"已加载引擎: {engine_file_path}")
        return engine
    except Exception as e:
        print(f"加载错误: {e}")
        return None

def build_trt(onnx_file, trt_file):
    
    if os.path.exists(trt_file):
        print("engine trt file detected, loading")
        engine = load_serialized_engine(trt_file)
        return engine
        
    # parse onnx and save trt file 
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) # trt.Logger.WARNING #INFO
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_file, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise ValueError("fail to parse onnx")
    
    config = builder.create_builder_config()
    if config is None:
        raise RuntimeError("Fail to build config")
    config.max_workspace_size = 1 << 30  # 设置工作空间大小
    config.set_flag(trt.BuilderFlag.FP16)  # 设置精度为 FP16 or FP32?
    
    try:
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Fail to build TensorRT engine")
    except Exception as e:
        print(f'bug: {str(e)}')
    
    with open(trt_file, "wb") as f:
        f.write(engine.serialize())
    print("Build engine successfully！")
    return engine


def get_bindings_info(engine): # wrong
    bindings = []
    for binding in engine:
        binding_shape = engine.get_binding_shape(binding)
        size = trt.volume(binding_shape) * engine.get_binding_dtype(binding).itemsize
        bindings.append({
            "name": binding,
            "index": engine.get_binding_index(binding),
            "shape": binding_shape,
            "dtype": np.dtype(trt.nptype(engine.get_binding_dtype(binding))),
            "size": size
        })
    return bindings

def check_gpu_memory():
    allocated = torch.cuda.memory_allocated()/1024**3
    reserved = torch.cuda.memory_reserved()/1024**3
    print(f'''[显存状态]
    已分配: {allocated:.2f}GB
    保留池: {reserved:.2f}GB
    可用: {torch.cuda.get_device_properties(0).total_memory/1024**3 - allocated:.2f}GB''')



# 获取输入输出绑定的索引和名称
# input_bindings = {}
# output_bindings = {}
# for idx in range(engine.num_bindings):
#     name = engine.get_binding_name(idx)
#     if engine.binding_is_input(idx):
#         input_bindings[name] = { #input
#             "index": idx,
#             "shape": engine.get_binding_shape(idx),
#             "dtype": engine.get_binding_dtype(idx)
#         }
#     else:
#         output_bindings[name] = { #output
#             "index": idx,
#             "shape": engine.get_binding_shape(idx),
#             "dtype": engine.get_binding_dtype(idx)
#         }




# import torch
# import onnxruntime as ort

# # 创建PyTorch GPU Tensor
# torch_input = torch.randn(1,3,224,224).cuda()

# # Step1: 转换PyTorch Tensor到DLPack格式
# dlpack = torch_input.to_dlpack()

# # Step2: 转换为ORT的OrtValue（零拷贝）
# ort_value = ort.OrtValue.from_dlpack(dlpack)

# # Step3: 执行推理
# ort_session = ort.InferenceSession("model.onnx", providers=['CUDAExecutionProvider'])
# outputs = ort_session.run(None, {'input': ort_value})
