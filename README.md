# Slimmable Neural Networks

[ICLR 2019 Paper](https://arxiv.org/abs/1812.08928) | [ArXiv](https://arxiv.org/abs/1812.08928) | [OpenReview](https://openreview.net/forum?id=H1gMCsAqY7) | [Detection](https://github.com/JiahuiYu/slimmable_networks/tree/detection) | [Model Zoo](#model-zoo) | [BibTex](#citing)

<img src="https://user-images.githubusercontent.com/22609465/50390872-1b3fb600-0702-11e9-8034-d0f41825d775.png" width=95%/>

Illustration of slimmable neural networks. The same model can run at different widths (number of active channels), permitting instant and adaptive accuracy-efficiency trade-offs.


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


## Model Zoo

| Model | Switches (Widths) | Top-1 Err. | MFLOPs | Model ID |
| :--- | :---: | :---: | ---: | :---: |
| S-MobileNet v1 | 1.00<br>0.75<br>0.50<br>0.25 | 28.5<br>30.5<br>35.2<br>46.9 | 569<br>325<br>150<br>41 | [a6285db](https://github.com/JiahuiYu/slimmable_networks/files/2709079/s_mobilenet_v1_0.25_0.5_0.75_1.0.pt.zip) |
| S-MobileNet v2 | 1.00<br>0.75<br>0.50<br>0.35 | 29.5<br>31.1<br>35.6<br>40.3 | 301<br>209<br>97<br>59 | [0593ffd](https://github.com/JiahuiYu/slimmable_networks/files/2709080/s_mobilenet_v2_0.35_0.5_0.75_1.0.pt.zip) |
| S-ShuffleNet | 2.00<br>1.00<br>0.50 | 28.6<br>34.5<br>42.8 | 524<br>138<br>38 | [1427f66](https://github.com/JiahuiYu/slimmable_networks/files/2709082/s_shufflenet_0.5_1.0_2.0.pt.zip) |
| S-ResNet-50 | 1.00<br>0.75<br>0.50<br>0.25 | 24.0<br>25.1<br>27.9<br>35.0 | 4.1G<br>2.3G<br>1.1G<br>278 | [3fca9cc](https://drive.google.com/open?id=1f6q37OkZaz_0GoOAwllHlXNWuKwor2fC) |


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
```
