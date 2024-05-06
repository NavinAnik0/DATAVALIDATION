# nfnet-training

This is an implementation of [nfnet](https://arxiv.org/abs/2102.06171) model where the repo mostly follows code from [this](https://github.com/benjs/nfnets_pytorch) repo.

## Description

This repo contains code for training the nfnet model and evaluating the saved models. The training code slightly differs from its predecessor as it loads the images from S3 and finally stores the model in S3. We are customizing the original nfnet model to achieve better results with our data. We have changed the original dataloading to use torchdata module. Evaluation of models is completely our own code, different from the predecessor repo.   

## Getting Started

### Installing

#### Option 1 
* Run the following .sh script, it will install cuda drivers, pip, a virtual environment and all the required packages. 
```
bash setup/prepare_ec2.sh setup
```

### Executing program

* You can run the following command to run training. 
  ```
  python train.py
  ```
* You can run the following command to run evaluation. 
   ```
  python evaluate_model.py
  ```
  
### Changing input for training and evaluation
* This repo contains a yaml files in *config_files* folder. Input parameter can be changed in corresponding yaml files for training and evaluation.


