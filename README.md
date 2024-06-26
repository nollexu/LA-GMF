# LA-GMF

PyTorch implementation of "Interpretable Medical Deep Framework by Logits-constraint Attention Guiding Graph-based Multi-scale Fusion for Alzheimer's Disease Analysis"

## Dependencies

* CUDA Version 12.2    
* torch 1.13.1+cu116
* torch-cluster 1.6.1+pt113cu116
* torch-geometric 2.1.0
* torch-scatter 2.1.1+pt113cu116
* torch-sparse 0.6.17+pt113cu116
* torch-spline-conv 1.2.2+pt113cu116
* torchaudio 0.13.1+cu116
* torchvision 0.14.1+cu116
* SimpleITK 2.2.1
* numpy 1.25.2
* Python 3.9.17
* pandas 2.0.3
* scikit-learn 1.3.0
## Usage

🐣For the hyperparameter λ(τ), we recommend delaying its assignment when using a smaller training set, and advancing its assignment when using a larger training set. For example, when only using ADNI1 for training, you can set λ=0(τ≤30)/λ=0.2(τ>30). When using ADNI1+ADNI2+ADNI3 for training, you can set λ=0(τ≤10)/λ=0.2(τ>10).

```plain
python main.py --device [GPU-id]
```

