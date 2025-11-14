CUDA_VISIBLE_DEVICES=0 \
python TeVNet/tev2hsv.py \
--weights-file yourpretraned \
--image-dir yourimg \
--smp_model Unet --smp_encoder resnet18 \
--output-dir output/hsv \
--vnums 4