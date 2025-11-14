CUDA_VISIBLE_DEVICES=0 \
python TeVNet/test.py \
--weights-file yourpretrained \
--image-dir yourimg \
--smp_model Unet --smp_encoder resnet18 \
--output-dir output/Tev_vnums4 \
--vnums 4