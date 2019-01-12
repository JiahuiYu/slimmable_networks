# Slimmable Neural Networks (detection experiments)

[ICLR 2019 Paper](https://arxiv.org/abs/1812.08928) | [ArXiv](https://arxiv.org/abs/1812.08928) | [OpenReview](https://openreview.net/forum?id=H1gMCsAqY7) | [Detection](https://github.com/JiahuiYu/slimmable_networks/tree/detection) | [Model Zoo](#model-zoo) | [BibTex](#citing)

<img src="https://user-images.githubusercontent.com/22609465/50390872-1b3fb600-0702-11e9-8034-d0f41825d775.png" width=95%/>

Illustration of slimmable neural networks. The same model can run at different widths (number of active channels), permitting instant and adaptive accuracy-efficiency trade-offs.


## Run

0. Requirements:
    * Setup according to [mmdetection](https://github.com/open-mmlab/mmdetection) based on commit [86e25e](https://github.com/open-mmlab/mmdetection/tree/86e25e41aea1c2170c0b242e486b1d4685134f31).
1. Testing:
    * Download our [pretrained models](https://drive.google.com/open?id=1ueqa1BYhDQ0ANm1j3NwrSTgNfKkPmrdJ) of mask-rcnn and faster-rcnn.
    * Choose the width multiplier to evaluate, where width ratio can be selected from {0.25, 0.5, 0.75, 1.0}.
    ```
    python tools/test.py configs/faster_rcnn_r50_fpn_1x.py {pretrained_model_file.pt} --eval bbox --width_mult {width_ratio}
    ```
2. Training:
    ```
    python -m torch.distributed.launch --nproc_per_node=8 tools/train.py configs/mask_rcnn_r50_fpn_1x.py --launcher pytorch
    ```
3. Still have questions?
    * If you still have questions, please search closed issues first. If the problem is not solved, please open a new.


## Technical Details

Implementing slimmable networks for detection is straightforward as shown in this single commit [8bc1dd8](https://github.com/JiahuiYu/slimmable_networks/commit/8bc1dd80490f8b8e3fd8ffe249c4fbf6dc4b9910).


## License

CC 4.0 Attribution-NonCommercial International

The software is for educaitonal and academic research purpose only.


## Acknowledgement

We would like to thank mmlab team for the detection toolbox mmdetection.


## Citing
```
@article{yu2018slimmable,
  title={Slimmable Neural Networks},
  author={Yu, Jiahui and Yang, Linjie and Xu, Ning and Yang, Jianchao and Huang, Thomas S},
  journal={arXiv preprint arXiv:1812.08928},
  year={2018}
}
```
