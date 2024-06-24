import argparse
import torch
import cv2
import numpy as np
import os
from model import Generator
from torchvision.transforms.functional import to_tensor
from PIL import Image, ImageEnhance, ImageFilter

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def load_image(image_path, target_resolution=None):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if target_resolution is not None:
        h, w = img.shape[:2]
        new_height = target_resolution[1]
        new_width = int(w * (new_height / h))
        img = cv2.resize(img, (new_width, new_height))

    img = torch.from_numpy(img)
    img = img / 127.5 - 1.0
    return img

def post_process_image(img: Image.Image) -> Image.Image:
    # Apply enhancements
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.2)  # Increase contrast

    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(1.5)  # Increase sharpness

    # Apply a slight Gaussian blur to reduce noise
    img = img.filter(ImageFilter.GaussianBlur(radius=1))

    return img

def test(args):
    device = args.device
    
    net = Generator()
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    net.to(device).eval()
    print(f"Model loaded: {args.checkpoint}")
    
    os.makedirs(args.output_dir, exist_ok=True)

    for image_name in sorted(os.listdir(args.input_dir)):
        if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
            continue
            
        # Set a higher resolution for the generated images with increased height
        image = load_image(os.path.join(args.input_dir, image_name), target_resolution=(1024, 1024))

        with torch.no_grad():
            input = image.permute(2, 0, 1).unsqueeze(0).to(device)
            out = net(input, args.upsample_align).squeeze(0).permute(1, 2, 0).cpu().numpy()
            out = (out + 1) * 127.5
            out = np.clip(out, 0, 255).astype(np.uint8)

        # Convert the output to PIL Image
        output_img = Image.fromarray(out)

        # Post-process the image
        output_img = post_process_image(output_img)
        
        # Save the post-processed image
        output_img.save(os.path.join(args.output_dir, image_name))
        print(f"Image saved: {image_name}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./pytorch_generator_Paprika.pt')
    parser.add_argument('--input_dir', type=str, default='./samples/inputs')
    parser.add_argument('--output_dir', type=str, default='./samples/results')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--upsample_align', type=bool, default=True)  # Enable upsample alignment
    args = parser.parse_args()
    
    test(args)
