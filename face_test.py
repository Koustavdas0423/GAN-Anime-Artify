import os
import dlib
import numpy as np
import argparse
import cv2
import torch
from typing import Union
from PIL import Image, ImageEnhance
import PIL  # Added import statement for PIL
from torchvision.transforms.functional import to_tensor, to_pil_image
from model import Generator
import scipy.ndimage

def face2paint(
    img: Image.Image,
    size: int,
    side_by_side: bool = True,
) -> Image.Image:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_fname = "C:\\AnimeGANv2\\models\\fine_tuned_generator.pth"
    model = Generator().eval().to(device)
    model.load_state_dict(torch.load(model_fname, map_location=device))

    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((size, size), Image.LANCZOS)

    input = to_tensor(img).unsqueeze(0) * 2 - 1
    output = model(input.to(device)).cpu()[0]

    if side_by_side:
        output = torch.cat([input[0], output], dim=2)

    output = (output * 0.5 + 0.5).clip(0, 1)

    return to_pil_image(output)

def get_dlib_face_detector(predictor_path: str = "shape_predictor_68_face_landmarks.dat"):
    detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(predictor_path)

    def detect_face_landmarks(img: Union[Image.Image, np.ndarray]):
        if isinstance(img, Image.Image):
            img = np.array(img)
        faces = []
        dets = detector(img)
        for d in dets:
            shape = shape_predictor(img, d)
            faces.append(np.array([[v.x, v.y] for v in shape.parts()]))
        return faces
    
    return detect_face_landmarks

def align_and_crop_face(
    img: Image.Image,
    landmarks: np.ndarray,
    expand: float = 1.3,  # Increased expand factor for better face cropping
    output_size: int = 1024, 
    transform_size: int = 4096,
    enable_padding: bool = True,
):
    lm = landmarks
    lm_chin = lm[0:17]
    lm_eyebrow_left = lm[17:22]
    lm_eyebrow_right = lm[22:27]
    lm_nose = lm[27:31]
    lm_nostrils = lm[31:36]
    lm_eye_left = lm[36:42]
    lm_eye_right = lm[42:48]
    lm_mouth_outer = lm[48:60]
    lm_mouth_inner = lm[60:68]

    eye_left = np.mean(lm_eye_left, axis=0)
    eye_right = np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_left = lm_mouth_outer[0]
    mouth_right = lm_mouth_outer[6]
    mouth_avg = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    x *= expand
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w - 1 - x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h - 1 - y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.LANCZOS)

    return img

def load_image(image_path, x32=False):
    img = cv2.imread(image_path).astype(np.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    if x32:
        def to_32s(x):
            return 256 if x < 256 else x - x % 32
        img = cv2.resize(img, (to_32s(w), to_32s(h)))

    img = torch.from_numpy(img)
    img = img / 127.5 - 1.0
    return img

def enhance_image(image: Image.Image) -> Image.Image:
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.5)  # Increase color saturation
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.3)  # Increase contrast
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(2.0)  # Increase sharpness
    return image

def test(args):
    model_fname = "face_paint_512_v2_0.pt"
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Generator().eval().to(device)
    model.load_state_dict(torch.load(model_fname, map_location=device))
    face_detector = get_dlib_face_detector()

    os.makedirs(args.output_dir, exist_ok=True)

    for image_name in sorted(os.listdir(args.input_dir)):
        if os.path.splitext(image_name)[-1].lower() not in [".jpg", ".png", ".bmp", ".tiff"]:
            continue

        image_path = os.path.join(args.input_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        print(f"Processing image: {image_path}")

        landmarks = face_detector(image)
        if not landmarks:
            print(f"No face landmarks found in image: {image_name}")
            continue
        
        for landmark in landmarks:
            face = align_and_crop_face(image, landmark, expand=1.3)
            output_image = face2paint(face, 1024)  # Increased resolution to 1024

            # Enhance the output image
            enhanced_image = enhance_image(output_image)

            enhanced_image.save(os.path.join(args.output_dir, image_name))
            print(f"Image saved: {os.path.join(args.output_dir, image_name)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./face_paint_512_v2_0.pt')
    parser.add_argument('--input_dir', type=str, default='./samples/inputs')
    parser.add_argument('--output_dir', type=str, default='./samples/results')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--upsample_align', type=bool, default=False)
    parser.add_argument('--x32', action="store_true")
    args = parser.parse_args()

    test(args)
