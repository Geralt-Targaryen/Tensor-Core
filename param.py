import torch
from torchvision import models
from array import array

def write_conv2d(layer, f):
    f.extend(list(torch.flatten(layer.weight.data).numpy()))

def write_batchnorm2d(layer, f):
    f.extend(list(torch.flatten(layer.running_mean.data).numpy()))
    f.extend(list(torch.flatten(layer.running_var.data).numpy()))
    f.extend(list(torch.flatten(layer.weight.data).numpy()))
    f.extend(list(torch.flatten(layer.bias.data).numpy()))

def write_block(block, f, down=False):
    write_conv2d(block.conv1, f)
    write_batchnorm2d(block.bn1, f)
    write_conv2d(block.conv2, f)
    write_batchnorm2d(block.bn2, f)
    if down:
        write_conv2d(block.downsample[0], f)
        write_batchnorm2d(block.downsample[1], f)

def write_linear(layer: torch.nn.Linear, f):
    f.extend(list(torch.flatten(layer.weight.data).numpy()))
    f.extend(list(torch.flatten(layer.bias.data).numpy()))

model = models.resnet18(pretrained=True)
model.eval()
f = []
write_conv2d(model.conv1, f)
write_batchnorm2d(model.bn1, f)
write_block(model.layer1[0], f, down=False)
write_block(model.layer1[1], f, down=False)
write_block(model.layer2[0], f, down=True)
write_block(model.layer2[1], f, down=False)
write_block(model.layer3[0], f, down=True)
write_block(model.layer3[1], f, down=False)
write_block(model.layer4[0], f, down=True)
write_block(model.layer4[1], f, down=False)
write_linear(model.fc, f)

with open('input/param.bin', 'wb') as f_param:
    param_array = array('f', f)
    param_array.tofile(f_param)
