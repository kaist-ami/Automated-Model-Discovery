# Automated Model Discovery via Multi-modal & Multi-step Pipeline
### [Project Page](https://kim-youwang.github.io/model-discovery) | [Paper](https://arxiv.org/abs/2509.25946)
[NeurIPS'25] Official Repository for 'Automated Model Discovery via Multi-modal & Multi-step Pipeline'

## Highlights

### Environment Setup
This code was developed at Ubuntu 20.04, using python=3.11, GPy==1.13.2, and GPy-ABCD==1.2.3.
Later versions should work, but it have not been tested.
```bash
conda create -n amd python=3.11
conda activate amd
pip install -r requirements.txt
```

## Getting Started with Kernel Structure Discovery
Since our model discovery is done in training-free method, you can run the code 

### Dataset Preparation
Univariate Time-series Dataset are already provided in `Automated-Model-Discovery/data`, in csv file. They are originated from `https://github.com/jamesrobertlloyd/gpss-research.git`.

### Running the Code
You can start with the code by simply executing:
```bash
python3 main_gp.py --noise 0 --data 1 --set_se_const True --model gpt-4o-mini
```

## Getting Started with Symbolic Regression

### Dataset Preparation
You can download dataset for Symbolic Regression at `https://github.com/merlerm/In-Context-Symbolic-Regression.git/data`, and please place them at `Automated-Model-Discovery/data/symbolic_regression`.

### Running the Code
You can start with the code by simply executing:
```bash
python3 main_sr.py experiment/function=nguyen/nguyen1
```

## Citation
If you find our code or paper helps, please consider citing our paper:
````BibTeX
@inproceedings{
jung-mok2025automated,
title={Automated Model Discovery via Multi-modal \& Multi-step Pipeline},
author={Lee Jung-Mok and Nam Hyeon-Woo and Moon Ye-Bin and Junhyun Nam and Tae-Hyun Oh},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=qGFvTIMS3W}
}
````

## Acknowledgement
We sincerly thank for the authors of these project for making their work publicly available:
- [GPy: Gaussian process framework in Python](https://github.com/SheffieldML/GPy.git)
- [GPy-ABCD: A Configurable Automatic Bayesian Covariance Discovery Implementation](https://github.com/T-Flet/GPy-ABCD)
- [In-Context-Symbolic-Regression: Leveraging Large Language Models for Function Discovery](https://github.com/merlerm/In-Context-Symbolic-Regression.git)