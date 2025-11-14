import numpy as np
import cv2
import torch
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio

def calculate_psnr(img1, img2):
    return peak_signal_noise_ratio(img1, img2)


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
    

def calculate_lpips(img1, img2, lpips_model):
    if len(img1.shape) == 2:
        image1_tensor = torch.tensor(np.array(img1)).unsqueeze(0).float() / 255.0
        image2_tensor = torch.tensor(np.array(img2)).unsqueeze(0).float() / 255.0
    else:
        image1_tensor = torch.tensor(np.array(img1)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image2_tensor = torch.tensor(np.array(img2)).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    distance = lpips_model(image1_tensor, image2_tensor)

    return distance.item()