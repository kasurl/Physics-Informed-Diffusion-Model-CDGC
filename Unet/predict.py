import cv2
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from model import UNet
import argparse


def predict_cracks(model, image_path, save_dir='results', threshold=0.4):
    """Predict cracks for single image"""
    os.makedirs(save_dir, exist_ok=True)

    img = Image.open(image_path).convert("L")
    original_size = img.size

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    img_tensor = transform(img)
    img_tensor = img_tensor.unsqueeze(0).cuda()

    with torch.no_grad():
        output = model(img_tensor)

    pred_mask = (output > threshold).float().squeeze().cpu().numpy()
    pred_mask = cv2.resize(pred_mask, original_size[::-1])
    pred_mask = (pred_mask * 255).astype(np.uint8)

    save_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(save_path, pred_mask)

    return save_path


def predict_folder(model, input_dir, output_dir, threshold=0.4):
    """Predict cracks for all images in folder"""
    os.makedirs(output_dir, exist_ok=True)

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    images_list = [f for f in os.listdir(input_dir) if f.lower().endswith(image_extensions)]

    if not images_list:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(images_list)} images in {input_dir}")

    for i, image in enumerate(images_list):
        img_path = os.path.join(input_dir, image)
        saved_path = predict_cracks(model, img_path, output_dir, threshold)
        print(f"[{i + 1}/{len(images_list)}] Processed: {saved_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict cracks using trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input_dir', type=str, required=True, help='Input images directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for predictions')
    parser.add_argument('--threshold', type=float, default=0.4, help='Prediction threshold')

    args = parser.parse_args()

    # Load model
    model = UNet().cuda()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    print(f"Model loaded from: {args.model_path}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Prediction threshold: {args.threshold}")

    # Predict all images in folder
    predict_folder(model, args.input_dir, args.output_dir, args.threshold)
    print("Prediction completed!")

