import cv2
import numpy as np
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    # User settings
    img_path = "../dataset/real/visible/140.png"
    gamma_r, gamma_g, gamma_b = 0.5, 1.0, 1.8
    fusion_alpha = 0.2

    # Load image
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Processing pipeline
    orig = img_rgb.copy()
    gamma_img = apply_manual_gamma(orig, gamma_r, gamma_g, gamma_b)
    normalized = dynamic_normalize(gamma_img)
    fused = adaptive_fusion(orig, normalized, alpha=fusion_alpha)

    # Display results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(orig)
    axs[0].set_title("Original IR")
    axs[0].axis("off")

    axs[1].imshow(gamma_img)
    axs[1].set_title(f"Gamma Enhanced\n(R={gamma_r}, G={gamma_g}, B={gamma_b})")
    axs[1].axis("off")

    axs[2].imshow(fused)
    axs[2].set_title(f"Fusion Result (α={fusion_alpha})")
    axs[2].axis("off")

    plt.tight_layout()
    plt.show()