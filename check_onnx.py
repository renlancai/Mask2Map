import argparse
import onnx
import onnxsim

def parse_args():
    parser = argparse.ArgumentParser(description='check onnx file')
    parser.add_argument('onnxfile', help='onnx file')
    args = parser.parse_args()
    return args


def main():
    print("hello world!")
    args = parse_args()
    onnxfile = args.onnxfile
    onnx_model = onnx.load(onnxfile)
    print(onnxfile)
    print(onnx_model)
    onnx.checker.check_model(onnx_model)
    #print(onnx.helper.printable_graph(onnx_model.graph))
    #
    model_onnx, check = onnxsim.simplify(onnx_model, check_n=3, skip_shape_inference=True)
    assert check, 'assert check failed'
    onnx.save(model_onnx, onnxfile + ".simplified")


if __name__ == '__main__':
    main()

