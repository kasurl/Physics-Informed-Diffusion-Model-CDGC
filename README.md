# 

<img src="PID.png" alt="PID" style="zoom:50%;" />

## Environment

We recommend you to install the environment with environment.yaml. 

```bash
conda env create --file=environment.yml
```

## Datasets


## Checkpoint

## CDGC Test

```sh
bash shell/CDGC.sh
```
## Evaluation

Use the shellscript to evaluate. `indir` is the input directory of visible RGB images, `outdir` is the output directory of translated infrared images,  We prepare some RGB images in `dataset/CDGC` for quick evaluation.

```sh
bash shell/run_test.sh
```
###  Evaluate the preformance
```bash
bash shell/eval.sh
```

## Train

### Dataset preparation

Prepare corresponding RGB and infrared images with same names in two directories.

### Stage 1: Train TeVNet

```bash
cd TeVNet
bash shell/train.sh
```

### Stage 2: Train PID

Use the shellscript to train. It is recommended to use our pretrained model to accelerate the train process.

```bash
bash shell/run_train.sh
```


## Acknowledgements

Our code is built upon [LDM](https://github.com/CompVis/latent-diffusion) and [HADAR](https://github.com/FanglinBao/HADAR). We thank the authors for their excellent work.

