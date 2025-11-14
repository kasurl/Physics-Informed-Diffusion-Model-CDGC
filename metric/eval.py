import argparse
import core.metrics as Metrics
from PIL import Image
import numpy as np
import glob
import lpips
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p1', '--path1', type=str)
    parser.add_argument('-p2', '--path2', type=str)
    parser.add_argument('-s', '--size', type=int, default=512)

    args = parser.parse_args()
    real_names = list(glob.glob('{}/*.png'.format(args.path1)))
    if len(real_names) == 0:
        real_names = list(glob.glob('{}/*.jpg'.format(args.path1)))
        
    fake_names = list(glob.glob('{}/*.png'.format(args.path2)))
    if len(fake_names) == 0:
        fake_names = list(glob.glob('{}/*.jpg'.format(args.path2)))
    print(len(real_names), len(fake_names))
    assert len(real_names) == len(fake_names)
    
    real_names.sort()
    fake_names.sort()
    
    lpips_model = lpips.LPIPS(net="alex")

    avg_psnr = []
    avg_ssim = []
    avg_lpips = []
    idx = 0
    
    for rname, fname in tqdm(zip(real_names, fake_names)):
        real_image = Image.open(rname).convert('L')
        crop = min(real_image.size[0], real_image.size[1])
        h, w = real_image.size[1], real_image.size[0]
        real_image = real_image.crop(((w - crop) // 2, (h - crop) // 2, (w + crop) // 2, (h + crop) // 2))
        real_img = np.array(real_image.resize((args.size, args.size)))
        fake_img = np.array(Image.open(fname).convert('L').resize((args.size, args.size)))
        
        psnr = Metrics.calculate_psnr(fake_img, real_img)
        ssim = Metrics.calculate_ssim(fake_img, real_img)
        lpips = Metrics.calculate_lpips(fake_img, real_img, lpips_model)
        
        avg_psnr.append(psnr)
        avg_ssim.append(ssim)
        avg_lpips.append(lpips)
        
    
    print(idx)

    # log
    with open('./metrics.txt', 'a+') as f:
        f.write(args.path1 + '\n')
        f.write('# Validation # PSNR: mean: {:.4e}, std: {:.4e}'.format(np.mean(avg_psnr), np.std(avg_psnr)) + '\n')
        f.write('# Validation # SSIM: mean: {:.4e}, std: {:.4e}'.format(np.mean(avg_ssim), np.std(avg_ssim)) + '\n')
        f.write('# Validation # LPIPS: mean: {:.4e}, std: {:.4e}'.format(np.mean(avg_lpips), np.std(avg_lpips)) + '\n')
        
        print('# Validation # PSNR: mean: {:.4e}, std: {:.4e}'.format(np.mean(avg_psnr), np.std(avg_psnr)))
        print('# Validation # SSIM: mean: {:.4e}, std: {:.4e}'.format(np.mean(avg_ssim), np.std(avg_ssim)))
        print('# Validation # LPIPS: mean: {:.4e}, std: {:.4e}'.format(np.mean(avg_lpips), np.std(avg_lpips)))
