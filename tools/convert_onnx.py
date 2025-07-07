import torch
import torch.onnx
import onnx
import argparse
import os
import _init_paths
import models

def parse_args():
    parser = argparse.ArgumentParser(description='Convert PIDNet model to ONNX format')
    parser.add_argument('--model-type', type=str, default='pidnet-s',
                      help='pidnet-s, pidnet-m, pidnet-l')
    parser.add_argument('--input-path', type=str, default='output/cropline/pidnet_small_cropline/best.pt',
                      help='path to the input model file')
    parser.add_argument('--output-path', type=str, default='output/cropline/PIDNet.onnx',
                      help='path to save the ONNX model')
    parser.add_argument('--num-classes', type=int, default=2,
                      help='number of classes (2 for cropline)')
    parser.add_argument('--scale-factor', type=int, default=8,
                      help='scale factor for segmentation')
    parser.add_argument('--input-height', type=int, default=480,
                      help='input height for the model')
    parser.add_argument('--input-width', type=int, default=640,
                      help='input width for the model')
    parser.add_argument('--opset-version', type=int, default=11,
                      help='ONNX opset version')
    
    args = parser.parse_args()
    return args

def load_pretrained(model, pretrained):
    if not os.path.isfile(pretrained):
        print(f"Error: Model file {pretrained} not found!")
        return model
        
    pretrained_dict = torch.load(pretrained, map_location='cpu')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    msg = f'Loaded {len(pretrained_dict)} parameters!'
    print('Attention!!!')
    print(msg)
    print('Over!!!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    return model

def convert_to_onnx(args):
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Initialize the model
    print(f"Initializing {args.model_type} model with {args.num_classes} classes...")
    model = models.pidnet.get_pred_model(args.model_type, args.num_classes)
    
    # Load the pretrained weights
    print(f"Loading weights from {args.input_path}...")
    model = load_pretrained(model, args.input_path)
    model.eval()
    
    # Create dummy input tensor with the specified dimensions
    dummy_input = torch.randn(1, 3, args.input_height, args.input_width)
    
    # Export the model to ONNX format
    print(f"Exporting model to {args.output_path}...")
    torch.onnx.export(model, 
                     dummy_input, 
                     args.output_path,
                     export_params=True,
                     opset_version=args.opset_version,
                     do_constant_folding=True,
                     input_names=['input'],
                     output_names=['output'],
                     dynamic_axes={'input': {0: 'batch_size'},
                                  'output': {0: 'batch_size'}})
    
    # Verify the ONNX model
    print("Verifying ONNX model...")
    try:
        onnx_model = onnx.load(args.output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
        print(f"Model exported to: {args.output_path}")
    except Exception as e:
        print(f"Error verifying ONNX model: {e}")

if __name__ == "__main__":
    args = parse_args()
    convert_to_onnx(args)
