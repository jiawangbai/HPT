<img src="https://github.com/jiawangbai/HPT/blob/main/misc/eccv.png" width="200" height="100"/><br/>
# Hardly Perceptible Trojan Attack (HPT)

Implementation of "Hardly Perceptible Trojan Attack against Neural Networks with Bit Flips", accepted to ECCV2022.

<img src="https://github.com/jiawangbai/HPT/blob/main/misc/pipeline.png" width="700" height="200"/><br/>

# Usage
First, clone this repository and download the weights of the victim model;

Then, set the data path in "config.json";

Finally, run the below command to attack 8-bit quantized ResNet-18 on CIFAR-10 with the default setting:

```shell
sh run.sh
```
