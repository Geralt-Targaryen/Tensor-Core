import torch
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
from torchmetrics import Accuracy
from array import array
import os

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
bs: int = 50

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder('data', transform=transform)
dataloader = DataLoader(dataset, batch_size=bs, shuffle=False)

model = models.resnet18(pretrained=True)
model.to(device)
model.eval()

data, labels = [], []
ACC = Accuracy(num_classes=1000)

for step, (x, y) in enumerate(dataloader):
    # each batch is one class
    with torch.no_grad():
        output = model(x.to(device)).argmax(dim=1).cpu()
        counts = torch.bincount(output)
        label = int(counts.argmax())
        ACC(output, torch.tensor([label]*bs))
        print(f'Step {step} Class {label}: %.4f' % (output.eq(label).int().sum() / bs))

        labels.extend([label]*bs)
        for sample in x:
            data.extend(list(torch.flatten(sample).numpy()))
        with open(f'input/data{step}.bin', 'wb') as f_data:
            data_array = array('f', data)
            data_array.tofile(f_data)
            data = []


with open('input/label.bin', 'wb') as f_label:
    label_array = array('i', labels)
    label_array.tofile(f_label)

for i in range(len(dataloader)):
    os.system(f'cat input/data{i}.bin >> input/data.bin')
    os.system(f'rm input/data{i}.bin')

acc = ACC.compute()
print("Avg acc: %.4f" % acc)
