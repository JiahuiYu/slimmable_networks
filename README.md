# Slimmable Networks

![version](https://img.shields.io/badge/version-v3.0.0--alpha-green.svg?style=plastic)
![pytorch](https://img.shields.io/badge/pytorch-v1.0.0-green.svg?style=plastic)
![license](https://img.shields.io/badge/license-CC_BY--NC-green.svg?style=plastic)


An open source framework for slimmable training on tasks of ImageNet classification and COCO detection, which has enabled numerous projects. <sup>[1](#snets), [2](#usnets), [3](#autoslim)</sup>

<strong  id="snets">1. Slimmable Neural Networks</strong> <sub> [ICLR 2019 Paper](https://arxiv.org/abs/1812.08928) | [OpenReview](https://openreview.net/forum?id=H1gMCsAqY7) | [Detection](https://github.com/JiahuiYu/slimmable_networks/tree/detection) | [Model Zoo](#slimmable-model-zoo)</sub>

<img src="https://user-images.githubusercontent.com/22609465/50390872-1b3fb600-0702-11e9-8034-d0f41825d775.png" width=95%/>

Illustration of slimmable neural networks. The same model can run at different widths (number of active channels), permitting instant and adaptive accuracy-efficiency trade-offs.
</div>


<strong id="usnets">2. Universally Slimmable Networks and Improved Training Techniques</strong> <sub> [Preprint](https://arxiv.org/abs/1903.05134) | [Model Zoo](#slimmable-model-zoo)</sub>

<img src="https://user-images.githubusercontent.com/22609465/54562571-45b5ae00-4995-11e9-8984-49e32d07e325.png" width=60%/>

Illustration of universally slimmable networks. The same model can run at **arbitrary** widths.


<strong id="autoslim">3. AutoSlim: Towards One-Shot Architecture Search for Channel Numbers</strong> <sub> [Preprint](https://arxiv.org/abs/1903.11728) | [Model Zoo](#slimmable-model-zoo)</sub>

<img src="https://user-images.githubusercontent.com/22609465/54886763-93309000-4e59-11e9-963a-c15bf49af3c0.gif" width=25%/><img src="https://user-images.githubusercontent.com/22609465/54886764-9592ea00-4e59-11e9-9541-924bbd9ff727.gif" width=25%/><img src="https://user-images.githubusercontent.com/22609465/54886766-97f54400-4e59-11e9-81bb-3b262df7c898.gif" width=25%/><img src="https://user-images.githubusercontent.com/22609465/54886768-9a579e00-4e59-11e9-9896-25e7eab7e2e0.gif" width=25%/>

AutoSlimming MobileNet v1, MobileNet v2, MNasNet and ResNet-50: the optimized number of channels under **each** computational budget (FLOPs).


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


## Slimmable Model Zoo

**[Slimmable Neural Networks](https://arxiv.org/abs/1812.08928)**


| Model | Switches (Widths) | Top-1 Err. | FLOPs | Model ID |
| :--- | :---: | :---: | ---: | :---: |
| S-MobileNet v1 | 1.00<br>0.75<br>0.50<br>0.25 | 28.5<br>30.5<br>35.2<br>46.9 | 569M<br>325M<br>150M<br>41M | [a6285db](https://github.com/JiahuiYu/slimmable_networks/files/2709079/s_mobilenet_v1_0.25_0.5_0.75_1.0.pt.zip) |
| S-MobileNet v2 | 1.00<br>0.75<br>0.50<br>0.35 | 29.5<br>31.1<br>35.6<br>40.3 | 301M<br>209M<br>97M<br>59M | [0593ffd](https://github.com/JiahuiYu/slimmable_networks/files/2709080/s_mobilenet_v2_0.35_0.5_0.75_1.0.pt.zip) |
| S-ShuffleNet | 2.00<br>1.00<br>0.50 | 28.6<br>34.5<br>42.8 | 524M<br>138M<br>38M | [1427f66](https://github.com/JiahuiYu/slimmable_networks/files/2709082/s_shufflenet_0.5_1.0_2.0.pt.zip) |
| S-ResNet-50 | 1.00<br>0.75<br>0.50<br>0.25 | 24.0<br>25.1<br>27.9<br>35.0 | 4.1G<br>2.3G<br>1.1G<br>278M | [3fca9cc](https://drive.google.com/open?id=1f6q37OkZaz_0GoOAwllHlXNWuKwor2fC) |


**[Universally Slimmable Networks and Improved Training Techniques](https://arxiv.org/abs/1903.05134)**

| Model | Model&#160;ID | Spectrum | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | |
| :- | :-: | :- | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - | - |
| US&#x2011;MobileNet&#160;v1 | [13d5af2](https://github.com/JiahuiYu/slimmable_networks/files/2979952/us_mobilenet_v1_calibrated.pt.zip) | Width<br>MFLOPs<br>Top-1 Err. | 1.0<br>568 <br>28.2  | 0.975 <br>543 <br>28.3  | 0.95 <br>517 <br>28.4  | 0.925 <br>490 <br>28.7  | 0.9 <br>466 <br>28.7  | 0.875 <br>443 <br>29.1  | 0.85 <br>421 <br>29.4  | 0.825 <br>389 <br>29.7  | 0.8 <br>366 <br>30.2  | 0.775 <br>345 <br>30.3  | 0.75 <br>325 <br>30.5  | 0.725 <br>306 <br>30.9  | 0.7 <br>287 <br>31.2  | 0.675 <br>267 <br>31.7  | 0.65 <br>249 <br>32.2  | 0.625 <br>232 <br>32.5  | 0.6 <br>217 <br>33.2  | 0.575 <br>201 <br>33.7  | 0.55 <br>177 <br>34.4  | 0.525 <br>162 <br>35.0  | 0.5 <br>149 <br>35.8  | 0.475 <br>136 <br>36.5  | 0.45 <br>124 <br>37.3  | 0.425 <br>114 <br>38.1  | 0.4 <br>100 <br>39.0  | 0.375 <br>89 <br>40.0  | 0.35 <br>80 <br>41.0  | 0.325 <br>71 <br>41.9  | 0.3 <br>64 <br>42.7  | 0.275 <br>48 <br>44.2  | 0.25<br>41<br>44.3 |
| US&#x2011;MobileNet&#160;v2 | [3880cad](https://github.com/JiahuiYu/slimmable_networks/files/2979953/us_mobilenet_v2_calibrated.pt.zip) | Width<br>MFLOPs<br>Top-1 Err. | 1.0 <br>300 <br>28.5 | 0.975 <br>299 <br>28.5 | 0.95 <br>284 <br>28.8 | 0.925 <br>274 <br>28.9 | 0.9 <br>269 <br>29.1 | 0.875 <br>268 <br>29.1 | 0.85 <br>254 <br>29.4 | 0.825 <br>235 <br>29.9 | 0.8 <br>222 <br>30.0 | 0.775 <br>213 <br>30.2 | 0.75 <br>209 <br>30.4 | 0.725 <br>185 <br>30.7 | 0.7 <br>173 <br>31.1 | 0.675 <br>165 <br>31.4 | 0.65 <br>161 <br>31.7 | 0.625 <br>161 <br>31.7 | 0.6 <br>151 <br>32.4 | 0.575 <br>150 <br>32.4 | 0.55 <br>106 <br>34.4 | 0.525 <br>100 <br>34.6 | 0.5 <br>97 <br>34.9 | 0.475 <br>96 <br>35.1 | 0.45 <br>88 <br>35.8 | 0.425 <br>88 <br>35.8 | 0.4 <br>80 <br>36.6 | 0.375 <br>80 <br>36.7 | 0.35<br>59<br>37.7 |


**[AutoSlim: Towards One-Shot Architecture Search for Channel Numbers](https://arxiv.org/abs/1903.11728)**

| Model | Top-1 Err. | FLOPs | Model ID |
| :--- | :---: | ---: | :---: |
| AutoSlim-MobileNet v1 | 27.0<br>28.5<br>32.1 | 572M<br>325M<br>150M | [coming soon]() |
| AutoSlim-MobileNet v2 | 24.6<br>25.8<br>27.0 | 505M<br>305M<br>207M | [coming soon]() |
| AutoSlim-MNasNet | 24.6<br>25.4<br>26.8 | 532M<br>315M<br>217M | [coming soon]() |
| AutoSlim-ResNet-50 | 24.0<br>24.4<br>26.0<br>27.8 | 3.0G<br>2.0G<br>1.0G<br>570M | [coming soon]() |


## Technical Details

Implementing slimmable networks and slimmable training is straightforward:
  * Switchable batchnorm and slimmable layers are implemented in [`models/slimmable_ops`](/models/slimmable_ops.py).
  * Slimmable training is implemented in [these lines](https://github.com/JiahuiYu/slimmable_networks/blob/aeb10c9f437208603145e073ee730f0d7dbfa80f/train.py#L281-L289) in [`train.py`](/train.py).


## License

CC 4.0 Attribution-NonCommercial International

The software is for educaitonal and academic research purpose only.
