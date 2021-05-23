# The Heterogeneity Hypothesis: Finding Layer-Wise Differentiated Network Architectures

This repository contains the PyTorch training and evaluation code for [LW-DNA models](https://arxiv.org/abs/2006.16242).

In this paper, we tackle the problem of convolutional neural network design. 
Instead of focusing on the design of the overall architecture, we investigate a design space that is usually overlooked, i.e. adjusting the channel configurations of predefined networks. 
We find that this adjustment can be achieved by shrinking widened baseline networks and leads to superior performance. 
Based on that, we articulate the heterogeneity hypothesis: with the same training protocol, 
there exists a layer-wise differentiated network architecture (LW-DNA) that can outperform the original network with regular channel configurations but with a lower level of model complexity.

The LW-DNA models are identified without extra computational cost or training time compared with the original network. 
This constraint leads to controlled experiments which direct the focus to the importance of layer-wise specific channel configurations. 
LW-DNA models come with advantages related to overfitting, i.e. the relative relationship between model complexity and dataset size. 
Experiments are conducted on various networks and datasets for image classification, visual tracking and image restoration. 
The resultant LW-DNA models consistently outperform the baseline models.


If you use this code for a paper please cite:

```
@article{li2020heterogeneity,
  title={The Heterogeneity Hypothesis: Finding Layer-Wise Dissimilated Network Architecture},
  author={Li, Yawei and Li, Wen and Danelljan, Martin and Zhang, Kai and Gu, Shuhang and Van Gool, Luc and Timofte, Radu},
  journal={arXiv preprint arXiv:2006.16242},
  year={2020}
}
```


# 1. Usage

## Data preparation

Download and extract [ImageNet](http://image-net.org/) train and val images.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/vision/stable/datasets.html#imagefolder) as follows:

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_18.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ......
│  ├── ......
```

## Prepare the code

I. Clone the repository locally:
```bash
git clone https://github.com/ofsoundof/Heterogeneity_Hypothesis
```
II. Install the required packages
```bash
pip install -r requirements.txt
```

# 2. Evaluation
To evaluate the performance of the derived LW-DNA models, please run the following command.

Please download the derived [LW-DNA model](https://drive.google.com/file/d/1eKm_qRiGo5cJmZN276M5-3ZL2tXkHZyB/view?usp=sharing) for ResNet50.


```bash
python main.py --save ResNet_ImageNet_L50 --template ImageNet --model ResNet_ImageNet --depth 50 --test_only \
--dir_save save/path \
--dir_data path/to/imagenet  \
--pretrain ../resnet50_lwdna.pt
```


# 3. Network Shrinkage and Training
To shrink a predefined network for ImageNet classification and train the derived LW-DNA model, please run the following command.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main_hh_grad.py --save checkpoint_name --template ImageNet --model ResNet_ImageNet_HH --batch_size 256 --epochs 150 --decay cosine \
--depth 50 --width_mult 2 --ratio 0.23 --remain_percentage 0.4 --linear_percentage 0.45 --grad_prune --epochs_grad 1 \
--dir_save save/path --dir_data path/to/imagenet --n_GPUs 4 --embedding_dim 2
```

# 4. Acknowledgements

This repository is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch). We thank the authors for making their EDSR codes public.

This repository is also based on the [implementation](https://github.com/ofsoundof/learning_filter_basis) of my former paper [Learning Filter Basis](https://arxiv.org/abs/1908.08932), [Group Sparsity](https://arxiv.org/abs/2003.08935), and [DHP](https://arxiv.org/abs/2003.13683). If you are interested, please refer to:

```
@inproceedings{li2019learning,
  title = {Learning Filter Basis for Convolutional Neural Network Compression},
  author = {Li, Yawei and Gu, Shuhang and Van Gool, Luc and Timofte, Radu},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
  year = {2019}
}

@inproceedings{li2020group,
  title={Group Sparsity: The Hinge Between Filter Pruning and Decomposition for Network Compression},
  author={Li, Yawei and Gu, Shuhang and Mayer, Christoph and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  year={2020}
}

@inproceedings{li2020dhp,
  title={DHP: Differentiable Meta Pruning via HyperNetworks},
  author={Li, Yawei and Gu, Shuhang and Zhang, Kai and Van Gool, Luc and Timofte, Radu},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020}
}
```
