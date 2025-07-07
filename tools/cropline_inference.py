import os
import cv2
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import _init_paths
import models
from datasets.cropline import CropLine

# Color map for cropline segmentation
COLOR_MAP = [[0, 0, 0], [0, 255, 0]]

def parse_args():
    parser = argparse.ArgumentParser(description='CropLine Segmentation Inference')
    parser.add_argument('--a', type=str, default='pidnet-s', 
                        help='model architecture: pidnet-s, pidnet-m, pidnet-l')
    parser.add_argument('--p', type=str, required=True, 
                        help='path to pretrained model')
    parser.add_argument('--r', type=str, default='../samples/', 
                        help='directory with input images')
    parser.add_argument('--t', type=str, default='.png', 
                        help='image file extension')
    parser.add_argument('--n', type=int, default=2, 
                        help='number of classes')
    parser.add_argument('--o', type=str, default='outputs', 
                        help='output directory name')
    return parser.parse_args()

def input_transform(image):
    # Normalize image for model input
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def load_pretrained(model, pretrained_path):
    if os.path.isfile(pretrained_path):
        print(f"Loading pretrained model: {pretrained_path}")
        pretrained_dict = torch.load(pretrained_path, map_location='cpu')
    else:
        print(f"Model file not found: {pretrained_path}")
        return model
        
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() 
                      if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
    print(f'Loaded {len(pretrained_dict)} parameters')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    return model

def main():
    args = parse_args()
    
    # Find all images in the input directory
    images_list = glob.glob(os.path.join(args.r, f'*{args.t}'))
    output_path = os.path.join(args.r, args.o)
    
    # Create output directory
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Load and prepare the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.pidnet.get_pred_model(args.a, args.n)
    model = load_pretrained(model, args.p)
    model = model.to(device)
    model.eval()
    
    print(f"Processing {len(images_list)} images")
    
    with torch.no_grad():
        for img_path in images_list:
            img_name = os.path.basename(img_path)
            print(f"Processing: {img_name}")
            
            # Read and preprocess the image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
                
            orig_size = img.shape
            result_img = np.zeros_like(img, dtype=np.uint8)
            
            # Transform image for model input
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).to(device)
            
            # Run inference
            pred = model(img)
            print(f"Model output size: {pred.size()}")
            pred = F.interpolate(pred, size=img.size()[-2:], 
                               mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            
            # Create visualization
            for i, color in enumerate(COLOR_MAP):
                for j in range(3):
                    result_img[:,:,j][pred==i] = COLOR_MAP[i][j]
            
            # Save result
            result_img = Image.fromarray(result_img)
            save_path = os.path.join(output_path, img_name)
            result_img.save(save_path)
            
    print(f"Results saved to {output_path}")

if __name__ == '__main__':
    main()
