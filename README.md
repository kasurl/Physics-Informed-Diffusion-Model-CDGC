# Physics-Informed-Diffusion-Model-CDGC
This repo is the official implementation of "Visible-to-Infrared Domain Translation of Pavement Crack Images by A Physics-Informed Diffusion Model"
<img src="Physics-Informed-Diffusion.png" alt="Physics-Informed-Diffusion" style="zoom:50%;" />

## Environment

We recommend you to install the environment with environment.yml. 

```bash
conda env create --file=environment.yml
```

## Datasets

The datasets for testing the physics-informed diffusion model to generate synthetic infrared images can be downloaded at: [Infra_val](https://1drv.ms/u/c/a2bda927b647e020/EQTln3WQxTpOoMAPPBfvaf8B7IfyIdeM3oawMQK71c09pw?e=nls6ap)

The datasets for testing the segmentation model can be downloaded at: [Seg Test](https://1drv.ms/u/c/a2bda927b647e020/ETz58PvjweBKjv8cCCnGhZgBX0N5OxBbD5SLyiYVsK11mw?e=cew6OL)

## Checkpoint

The pretrained physics-informed diffusion model for generating synthetic infrared images can be downloaded at: [Physics-Informed Diffusion Model](https://1drv.ms/u/c/a2bda927b647e020/ESLMwCkEsYZCvwdLfYG0Hp8B1HvcXuf0MuSA1vkAZHqZNg?e=45y4Uj)

The pretrained segmentation model for testing the crack segmentation performance can be downloaded at: [Unet model](https://1drv.ms/u/c/a2bda927b647e020/Ec7o6skjXh9NqDQqmZb3du4BHOp2jatTrTEKAjysvcu_Mw?e=2kCXRR)

## CDGC Test
Use the shellscript to present the performance of CDGC module
```sh
bash shell/CDGC.sh
```
## Synthetic infrared imgs Evalutation

Use the shellscript to evaluate. `indir` is the input directory of visible RGB images, `outdir` is the output directory of translated infrared images.

```sh
bash shell/run_test.sh
```

###  Evaluate the preformance
```bash
bash shell/eval.sh
```

## Train Physics-Informed-Diffusion 

### Dataset preparation

Prepare corresponding RGB and infrared images with same names in two directories.

### Stage 1: TeVNet

#### Train TeVNet
```bash
bash TeVNet/shell/train.sh
```

#### Test TeVNet or output TeV decomposition variables.

```sh
bash TeVNet/shell/test_vnums4.sh
```

#### TeV -> HSV images

```sh
bash TeVNet/shell/test_tev2hsv.sh
```

### Stage 2: Train AutoEncoder

Use the shellscript to train Autoencoder. 

```bash
bash shell/run_train_1st.sh
```

### Stage 3: Train Physics-Informed-Diffusion

Use the shellscript to train. 

```bash
bash shell/run_train.sh
```

## Unet
 
### Train

```bash
bash Unet/shell/train.sh
```

### Predict

```bash
bash Unet/shell/predict.sh
```

## Acknowledgements

Our code is built upon [LDM](https://github.com/CompVis/latent-diffusion) and [PID](https://github.com/fangyuanmao/PID). We thank the authors for their excellent work.

