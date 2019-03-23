# Slimmable Networks

An open source framework for slimmable training on tasks of ImageNet classification and COCO detection, which has enabled numerous projects<sup>[1](#snets), [2](#usnets)</sup>.

<strong  id="snets">1. Slimmable Neural Networks</strong> <sub> [ICLR 2019 Paper](https://arxiv.org/abs/1812.08928) | [OpenReview](https://openreview.net/forum?id=H1gMCsAqY7) | [Detection](https://github.com/JiahuiYu/slimmable_networks/tree/detection) | [Model Zoo](/MODEL_ZOO.md) | [BibTex](#citing) </sub>

<img src="https://user-images.githubusercontent.com/22609465/50390872-1b3fb600-0702-11e9-8034-d0f41825d775.png" width=90%/>

Illustration of slimmable neural networks. The same model can run at different widths (number of active channels), permitting instant and adaptive accuracy-efficiency trade-offs.
</div>

<strong id="usnets">2. Universally Slimmable Networks and Improved Training Techniques</strong> <sub> [Preprint](https://arxiv.org/abs/1903.05134) | [Model Zoo](/MODEL_ZOO.md) | [BibTex](#citing) </sub>

<img src="https://user-images.githubusercontent.com/22609465/54562571-45b5ae00-4995-11e9-8984-49e32d07e325.png" width=50%/>

Illustration of universally slimmable networks. The same model can run at **arbitrary** widths.


## Run

0. Requirements:
    * python3, pytorch 1.0, torchvision 0.2.1, pyyaml 3.13.
    * Prepare ImageNet-1k data following pytorch [example](https://github.com/pytorch/examples/tree/master/imagenet).
1. Training and Testing:
    * The codebase is a general ImageNet training framework using yaml config under `apps` dir, based on PyTorch.
    * To test, download pretrained models to `logs` dir and directly run command.
    * To train, comment `test_only` and `pretrained` in config file. You will need to manage [visible gpus](https://devblogs.nvidia.com/cuda-pro-tip-control-gpu-visibility-cuda_visible_devices/) by yourself.
    * Command: `python train.py app:{apps/***.yml}`. `{apps/***.yml}` is config file. Do not miss `app:` prefix.
    * Training and testing of MSCOCO benchmarks are released under branch [detection](https://github.com/JiahuiYu/slimmable_networks/tree/detection).
2. Still have questions?
    * If you still have questions, please search closed issues first. If the problem is not solved, please open a new.


## Technical Details

Implementing slimmable networks and slimmable training is straightforward:
  * Switchable batchnorm and slimmable layers are implemented in [`models/slimmable_ops`](/models/slimmable_ops.py).
  * Slimmable training is implemented in [these lines](https://github.com/JiahuiYu/slimmable_networks/blob/aeb10c9f437208603145e073ee730f0d7dbfa80f/train.py#L281-L289) in [`train.py`](/train.py).


## License

CC 4.0 Attribution-NonCommercial International

The software is for educaitonal and academic research purpose only.


## Citing
```
@article{yu2018slimmable,
  title={Slimmable Neural Networks},
  author={Yu, Jiahui and Yang, Linjie and Xu, Ning and Yang, Jianchao and Huang, Thomas S},
  journal={arXiv preprint arXiv:1812.08928},
  year={2018}
}

@article{yu2019universally,
  title={Universally Slimmable Networks and Improved Training Techniques},
  author={Yu, Jiahui and Huang, Thomas},
  journal={arXiv preprint arXiv:1903.05134},
  year={2019}
}
```
