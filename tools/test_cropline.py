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

# Define the color map for cropline segmentation (black for background, green for cropline)
color_map = [[0, 0, 0], [0, 255, 0]]

def parse_args():
    parser = argparse.ArgumentParser(description='Inference for CropLine segmentation')
    parser.add_argument('--a', '-a', type=str, default='pidnet-s', 
                        help='architecture (default: pidnet-s)')
    parser.add_argument('--p', '-p', type=str, default='', 
                        help='path to the pretrained model')
    parser.add_argument('--r', '-r', type=str, default='../samples/', 
                        help='root path of the images')
    parser.add_argument('--t', '-t', type=str, default='.png', 
                        help='file extension of the images')
    parser.add_argument('--n', '-n', type=int, default=2, 
                        help='number of classes (default: 2 for cropline)')
    parser.add_argument('--o', '-o', type=str, default='outputs', 
                        help='output directory name')
    args = parser.parse_args()
    return args

def input_transform(image):
    # Normalize the image
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    image = image.astype(np.float32)[:, :, ::-1]
    image = image / 255.0
    image -= mean
    image /= std
    return image

def load_pretrained(model, pretrained):
    if os.path.isfile(pretrained):
        print(f"Loading pretrained model from {pretrained}")
        pretrained_dict = torch.load(pretrained, map_location='cpu')
    else:
        print(f"No pretrained model found at {pretrained}")
        return model
        
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

if __name__ == '__main__':
    args = parse_args()
    args.r = '/home/user/workspace/PIDNet/samples'
    args.t = '.png'
    args.o = '/home/user/workspace/PIDNet/samples/outputs'
    args.p = '/home/user/workspace/PIDNet/output/cropline/pidnet_small_cropline/best.pt'

    images_list = glob.glob(os.path.join(args.r, '*' + args.t))
    sv_path = os.path.join(args.r, args.o + '/')
    
    # Create output directory if it doesn't exist
    if not os.path.exists(sv_path):
        os.makedirs(sv_path)
    
    # Initialize model with the correct number of classes for cropline
    model = models.pidnet.get_pred_model(args.a, args.n)
    model = load_pretrained(model, args.p).cuda()
    model.eval()
    
    print(f"Found {len(images_list)} images for inference")
    
    with torch.no_grad():
        for img_path in images_list:
            img_name = os.path.basename(img_path)
            print(f"Processing image: {img_name}")
            
            # Read image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to read image: {img_path}")
                continue
                
            # Create output image
            sv_img = np.zeros_like(img).astype(np.uint8)
            
            # Preprocess image
            img = input_transform(img)
            img = img.transpose((2, 0, 1)).copy()
            img = torch.from_numpy(img).unsqueeze(0).cuda()
            
            # Run inference
            pred = model(img)
            pred = F.interpolate(pred, size=img.size()[-2:], mode='bilinear', align_corners=True)
            pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
            
            # Map prediction to colors
            for i, color in enumerate(color_map):
                for j in range(3):
                    sv_img[:,:,j][pred==i] = color_map[i][j]
            
            # Save result
            sv_img = Image.fromarray(sv_img)
            sv_img.save(os.path.join(sv_path, img_name))
            print(f"Saved result to {os.path.join(sv_path, img_name)}")
    
    print(f"Inference completed. Results saved to {sv_path}")
