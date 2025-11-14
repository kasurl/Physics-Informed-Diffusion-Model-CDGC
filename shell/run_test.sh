CUDA_VISIBLE_DEVICES=0 python scripts/rgb2ir_vqf8.py --steps 200 \
--indir Visible_Image \
--outdir Synthetic_IR \
--config configs/latent-diffusion/CDGC.yaml \
--checkpoint your_pretrained_model.ckpt