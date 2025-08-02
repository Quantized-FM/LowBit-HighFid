# Low-Bit, High-Fidelity: Optimal Transport Quantization for Flow Matching

This repo contains code necessary to replicate, reproduce and validate results from the paper "Low-Bit, High-Fidelity: Optimal Transport Quantization for Flow Matching," submitted to AAAI 2026. 

## Training Flow Matching Models
All models have been trained according to the official Flow Matching repository (accessible through this [link](https://github.com/facebookresearch/flow_matching). 5 FM models have been trained in total, corresponding to each of: 
- MNIST
- FashionMNIST
- CIFAR-10
- CelebA
- ImageNet
  
(In order of increasing data complexity)

Details for training have been included in the corresponding {dataset}-args.json files (found through ```./args```). 
