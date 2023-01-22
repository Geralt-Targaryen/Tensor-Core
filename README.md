# ResNet18 with CUDA Tensor Core

### prepare data
Move 100 ImageNet val class (each with 50 sampels) into ```data``` directory

Save preprocessed data  and labels in ```input```, and compute torchvision's ResNet18's acc
```
python torch_baseline.py
```
torch's acc: ~ 0.75.

### Save parameters
Save the parameters of torchvision's ResNet18 to ```input/param.bin```
```
python param.py
```

### Compile
```
nvcc test.cu matrix.cu im2col.cu conv.cu layers.cu utils.cu sim.cu -o test -arch=sm_70
```

### Inference
Run inference on 5000 images with batch size 16
```
./test 16
```

### Terminology
WMMA: warp-level matrix multiply and accumulate

GEMM: general matrix multiplication
