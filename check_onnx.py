import argparse
import onnx

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


if __name__ == '__main__':
    main()

