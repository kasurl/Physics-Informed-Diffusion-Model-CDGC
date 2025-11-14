CUDA_VISIBLE_DEVICES=0 \
python TeVNet/train.py \
--smp_model Unet --smp_encoder resnet18 --smp_encoder_weights imagenet \
--num-epochs 1000 \
--num-epochs-save 50 \
--num-epochs-val 10 \
--outputs-dir  outputs/TeVNet \
--batch-size 16 \
--lr 0.001 \
--train-dir yourtrain \
--eval-dir yourtest \
--vnums 4