![Python 3.6](https://img.shields.io/badge/python-3.6-green.svg)

# CP-GAN: Class-Distinct and Class-Mutual Image Generation with GANs (BMVC 2019)

This repository provides PyTorch implementation for **classifier's posterior GAN** ([**CP-GAN**](https://arxiv.org/abs/1811.11163)). Given *class-overlapping* data, CP-GAN can learn **a class-distinct and class-mutual image generator** that can capture between-class relationships and generate an image selectively on the basis of the *class specificity*.

<img src="docs/images/examples.png" width=100% alt="CP-GAN examples">

**NOTE:**
In CVPR 2019, we proposed **label-noise robust GAN** ([**rGAN**](https://arxiv.org/abs/1811.11165)) [1], which is another conditional extension of GAN for noisy labeled data. Check it from [**HERE**](https://takuhirok.github.io/rGAN/)!

## Paper

Class-Distinct and Class-Mutual Image Generation with GANs.<br>
[Takuhiro Kaneko](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/), [Yoshitaka Ushiku](https://yoshitakaushiku.net/), and [Tatsuya Harada](https://www.mi.t.u-tokyo.ac.jp/harada/).<br>
In BMVC 2019 (Spotlight).

[[Project]](https://takuhirok.github.io/CP-GAN/)
[[Paper]](https://arxiv.org/abs/1811.11163)
[[Slides]](docs/CP-GAN_slides.pdf)

## Overview

Our goal is, given *class-overlapping* data, to construct a **class-distinct and class-mutual image generator** that can selectively generate an image conditioned on the *class specificity*. To solve this problem, we propose **CP-GAN** (b), in which we redesign the generator input and the objective function of auxiliary classifier GAN (AC-GAN) [2]&nbsp;(a). Precisely, we employ the classifier’s posterior to represent the between-class relationships and incorporate it into the generator input. Additionally, we optimize the generator so that the classifier’s posterior of generated data corresponds with that of real data. This formulation allows CP-GAN to capture the between-class relationships in a data-driven manner and to generate an image conditioned on the *class specificity*.

<img src="docs/images/networks.png" width=100% alt="networks">

## Installation

Clone this repository.

```bash
git clone https://github.com/takuhirok/CP-GAN.git
cd CP-GAN/
```
First install Python 3.6 and then install [PyTorch](https://pytorch.org/) 1.0 and the other dependencies by

```bash
pip install -r requirements.txt
```

## Train

To train the model, run the following line.

```bash
python train.py \
    --dataset [cifar10/cifar10to5/cifar7to3] \
    --trainer [acgan/cpgan] \
    --out output_directory_path
```

Please choose one of the options in the square brackets ([ ]).

### Example
To train **CP-GAN** on **CIFAR-10to5**, run the following line.

```bash
python train.py \
    --dataset cifar10to5 \
    --trainer cpgan \
    --out outputs/cifar10to5/cpgan
```

## Test
To generate images, run the following line.

```bash
python test.py \
    --dataset [cifar10/cifar10to5/cifar7to3] \
    --g_path trained_model_path \
    --out output_directory_path
```

Please choose one of the options in the square brackets ([ ]).

### Example
To test the above-trained model on **CIFAR-10to5**, run the following line.

```bash
python test.py \
    --dataset cifar10to5 \
    --g_path outputs/cifar10to5/cpgan/netG_iter_100000.pth \
    --out samples
```

## Samples
### CIFAR-10to5
**Class-overlapping setting.**
The original **ten** classes (0,...,9; defined in (a)) are divided into **five** classes (*A*,...,*E*) with class overlapping, as shown in (b).

<img src="docs/images/overlap.png" width=100% alt="CIFAR-10to5 class-overlapping setting">

**Results.**
Each column shows samples associated with the same class-distinct and class-mutual states: *airplane*, *automobile*, *bird*, *cat*, *deer*, *dog*, *frog*, *horse*, *ship*, and *truck*, respectively, from left to right. Each row includes samples generated from a fixed ***z****<sup>g</sup>* and a varied ***y****<sup>g</sup>*. CP-GAN (b) succeeds in selectively generating class-distinct (red font) and class-mutual (blue font) images, whereas AC-GAN (a) fails to do so.

<img src="docs/images/samples.png" width=100% alt="CIFAR-10to5 samples">

## Citation
If you use this code for your research, please cite our paper.

```
@inproceedings{kaneko2019CP-GAN,
  title={Class-Distinct and Class-Mutual Image Generation with GANs},
  author={Kaneko, Takuhiro and Ushiku, Yoshitaka and Harada, Tatsuya},
  booktitle={Proceedings of the British Machine Vision Conference},
  year={2019}
}
```

## Related work

1. T. Kaneko, Y. Ushiku, T. Harada. Label-Noise Robust Generative Adversarial Networks, In CVPR, 2019. [[arXiv]](https://arxiv.org/abs/1811.11165) [[Project]](https://takuhirok.github.io/rGAN/) [[Code]](https://github.com/takuhirok/rGAN/)
2. A. Odena, C. Olah, and J. Shlens. Conditional image synthesis with auxiliary classifier GANs. In ICML, 2017. [[arXiv]](https://arxiv.org/abs/1610.09585)