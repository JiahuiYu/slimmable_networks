# Slimmable Model Zoo

## Slimmable Neural Networks ([ICLR 2019](https://arxiv.org/abs/1812.08928))


| Model | Switches (Widths) | Top-1 Err. | MFLOPs | Model ID |
| :--- | :---: | :---: | ---: | :---: |
| S-MobileNet v1 | 1.00<br>0.75<br>0.50<br>0.25 | 28.5<br>30.5<br>35.2<br>46.9 | 569<br>325<br>150<br>41 | [a6285db](https://github.com/JiahuiYu/slimmable_networks/files/2709079/s_mobilenet_v1_0.25_0.5_0.75_1.0.pt.zip) |
| S-MobileNet v2 | 1.00<br>0.75<br>0.50<br>0.35 | 29.5<br>31.1<br>35.6<br>40.3 | 301<br>209<br>97<br>59 | [0593ffd](https://github.com/JiahuiYu/slimmable_networks/files/2709080/s_mobilenet_v2_0.35_0.5_0.75_1.0.pt.zip) |
| S-ShuffleNet | 2.00<br>1.00<br>0.50 | 28.6<br>34.5<br>42.8 | 524<br>138<br>38 | [1427f66](https://github.com/JiahuiYu/slimmable_networks/files/2709082/s_shufflenet_0.5_1.0_2.0.pt.zip) |
| S-ResNet-50 | 1.00<br>0.75<br>0.50<br>0.25 | 24.0<br>25.1<br>27.9<br>35.0 | 4.1G<br>2.3G<br>1.1G<br>278 | [3fca9cc](https://drive.google.com/open?id=1f6q37OkZaz_0GoOAwllHlXNWuKwor2fC) |


## Universally Slimmable Networks and Improved Training Techniques ([Preprint](https://arxiv.org/abs/1903.05134))

| Model | Widths | Top-1 Err. | MFLOPs | Model ID |
| :--- | :--- | :---: | ---: | :---: |
| US-MobileNet v1 | 1.0<br> 0.975<br> 0.95<br> 0.925<br> 0.9<br> 0.875<br> 0.85<br> 0.825<br> 0.8<br> 0.775<br> 0.75<br> 0.725<br> 0.7<br> 0.675<br> 0.65<br> 0.625<br> 0.6<br> 0.575<br> 0.55<br> 0.525<br> 0.5<br> 0.475<br> 0.45<br> 0.425<br> 0.4<br> 0.375<br> 0.35<br> 0.325<br> 0.3<br> 0.275<br> 0.25 | 28.2<br> 28.3<br> 28.4<br> 28.7<br> 28.7<br> 29.1<br> 29.4<br> 29.7<br> 30.2<br> 30.3<br> 30.5<br> 30.9<br> 31.2<br> 31.7<br> 32.2<br> 32.5<br> 33.2<br> 33.7<br> 34.4<br> 35.0<br> 35.8<br> 36.5<br> 37.3<br> 38.1<br> 39.0<br> 40.0<br> 41.0<br> 41.9<br> 42.7<br> 44.2<br> 44.3 | 568<br> 543<br> 517<br> 490<br> 466<br> 443<br> 421<br> 389<br> 366<br> 345<br> 325<br> 306<br> 287<br> 267<br> 249<br> 232<br> 217<br> 201<br> 177<br> 162<br> 149<br> 136<br> 124<br> 114<br> 100<br> 89<br> 80<br> 71<br> 64<br> 48<br> 41 | [13d5af2](https://github.com/JiahuiYu/slimmable_networks/files/2979952/us_mobilenet_v1_calibrated.pt.zip) |
| US-MobileNet v2 | 1.0<br> 0.975<br> 0.95<br> 0.925<br> 0.9<br> 0.875<br> 0.85<br> 0.825<br> 0.8<br> 0.775<br> 0.75<br> 0.725<br> 0.7<br> 0.675<br> 0.65<br> 0.625<br> 0.6<br> 0.575<br> 0.55<br> 0.525<br> 0.5<br> 0.475<br> 0.45<br> 0.425<br> 0.4<br> 0.375<br> 0.35 | 28.5<br> 28.5<br> 28.8<br> 28.9<br> 29.1<br> 29.1<br> 29.4<br> 29.9<br> 30.0<br> 30.2<br> 30.4<br> 30.7<br> 31.1<br> 31.4<br> 31.7<br> 31.7<br> 32.4<br> 32.4<br> 34.4<br> 34.6<br> 34.9<br> 35.1<br> 35.8<br> 35.8<br> 36.6<br> 36.7<br> 37.7<br> | 300<br> 299<br> 284<br> 274<br> 269<br> 268<br> 254<br> 235<br> 222<br> 213<br> 209<br> 185<br> 173<br> 165<br> 161<br> 161<br> 151<br> 150<br> 106<br> 100<br> 97<br> 96<br> 88<br> 88<br> 80<br> 80<br> 59 | [3880cad](https://github.com/JiahuiYu/slimmable_networks/files/2979953/us_mobilenet_v2_calibrated.pt.zip) |
