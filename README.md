## Instructions
To load models from HF, please install the dependencies first
```python
!pip uninstall forgetting_transformer && pip install -U git+https://github.com/zhixuan-lin/forgetting-transformer
!pip install pytest einops numpy
!pip install torch==2.4.0
!pip install transformers==4.44.0
# No guarantee other commits would work; we may fix this later
!pip install --no-deps --force-reinstall git+https://github.com/sustcsonglin/flash-linear-attention.git@1c5937eeeb8b0aa17bed5ee6dae345b353196bd4
```