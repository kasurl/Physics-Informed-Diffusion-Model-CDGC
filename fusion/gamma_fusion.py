import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


def apply_manual_gamma(img_rgb, gamma_r=1.0, gamma_g=1.0, gamma_b=1.0):
    """Apply gamma correction to RGB channels independently"""
    x = img_rgb.astype(np.float32) / 255.0
    x[..., 0] = np.power(x[..., 0], gamma_r)
    x[..., 1] = np.power(x[..., 1], gamma_g)
    x[..., 2] = np.power(x[..., 2], gamma_b)
    x = np.clip(x, 0, 1)
    return (x * 255).astype(np.uint8)


def dynamic_normalize(img):
    """Normalize image using dynamic range adjustment"""
    img = img.astype(np.float32)
    min_v, max_v = np.percentile(img, (1, 99))
    img = np.clip((img - min_v) / (max_v - min_v + 1e-5), 0, 1)
    return (img * 255).astype(np.uint8)


def adaptive_fusion(orig, enhanced, alpha=0.5):
    """Fuse original and enhanced images with adaptive weighting"""
    fused = alpha * enhanced.astype(np.float32) + (1 - alpha) * orig.astype(np.float32)
    return np.clip(fused, 0, 255).astype(np.uint8)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Image enhancement with gamma correction and fusion')
    parser.add_argument('--input', '-i', required=True, help='Input image path')
    parser.add_argument('--gamma_r', type=float, default=0.5, help='Gamma value for red channel')
    parser.add_argument('--gamma_g', type=float, default=1.0, help='Gamma value for green channel')
    parser.add_argument('--gamma_b', type=float, default=1.8, help='Gamma value for blue channel')
    parser.add_argument('--alpha', type=float, default=0.2, help='Fusion alpha value')
    parser.add_argument('--output', '-o', help='Output image path (optional)')

    args = parser.parse_args()

    # Load image
    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {args.input}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Processing pipeline
    orig = img_rgb.copy()
    gamma_img = apply_manual_gamma(orig, args.gamma_r, args.gamma_g, args.gamma_b)
    normalized = dynamic_normalize(gamma_img)
    fused = adaptive_fusion(orig, normalized, alpha=args.alpha)

    # Display results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(orig)
    axs[0].set_title("Original IR")
    axs[0].axis("off")

    axs[1].imshow(gamma_img)
    axs[1].set_title(f"Gamma Enhanced\n(R={args.gamma_r}, G={args.gamma_g}, B={args.gamma_b})")
    axs[1].axis("off")

    axs[2].imshow(fused)
    axs[2].set_title(f"Fusion Result (α={args.alpha})")
    axs[2].axis("off")

    plt.tight_layout()

    # Save output if specified
    if args.output:
        # Convert fused image back to BGR for OpenCV saving
        fused_bgr = cv2.cvtColor(fused, cv2.COLOR_RGB2BGR)
        cv2.imwrite(args.output, fused_bgr)
        print(f"Output image saved to: {args.output}")

    plt.show()


if __name__ == "__main__":
    main()