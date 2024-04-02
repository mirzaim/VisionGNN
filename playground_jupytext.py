# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: myenv
#     language: python
#     name: python3
# ---

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
from dataset import ImageNetteDataset
from tqdm import tqdm
import time
from model import VGNN
from original_model import DeepGCN

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)

# %%
# %load_ext autoreload
# %autoreload 2

# %%
PATCH_SIZE = 16
BATCH_SIZE = 64
EPOCHS = 80
NUM_CLASSES = 10

# %%


# %%
class Classifier(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.backbone = VGNN()

        self.predictor = nn.Sequential(
            nn.Linear(320*196, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Linear(1024, n_classes)
        )
        self.fc1 = nn.Linear(320*196, 256)
        self.fc2 = nn.Linear(256, n_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        B, N, C = features.shape
        x = self.predictor(features.view(B, -1))
        return features, x

# %%
model = Classifier(n_classes=NUM_CLASSES)
model(torch.randn(32, 3, 224, 224))[-1].shape
model.to(device)
print('Model loaded')

# %%
# opt = {
#     'k': 9,
#     'act': 'gelu',
#     'norm': 'batch',
#     'bias': True,
#     'epsilon': 0.2,
#     'use_stochastic': False,
#     'conv': 'mr',
#     'n_blocks': 16,
#     'drop_path': 0.1,
#     'use_dilation': True,
#     'dropout': 0,
#     'n_classes': NUM_CLASSES,
#     'n_filters': 320,
# }
# opt = type('allMyFields', (object,), opt)

# model = DeepGCN(opt).to(device)

# %%

# %%
# with torch.no_grad():
#     x = torch.rand(32, 3, 224, 224).to(device)
#     print(model.stem(x).shape)

# %%
# nn.Sequential(
#     nn.Conv2d(3, 64, 1, 1, 0),
# )
# nn.Sequential(
#     nn.Flatten(),
#     nn.Linear(),
# )

# %%

# %%

# %%
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandAugment(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means,
                         std=pretrained_stds),
    transforms.RandomErasing(),
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means,
                         std=pretrained_stds),
])

cutmix = v2.CutMix(num_classes=NUM_CLASSES)
mixup = v2.MixUp(num_classes=NUM_CLASSES)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

train_dataset = ImageNetteDataset('data/imagenette2-320', split='train', transform=transform_train)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
val_dataset = ImageNetteDataset('data/imagenette2-320', split='val', transform=transform_val)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# %%
def train_step(model, dataloader, optimizer, criterion, device, epoch=None, mix_aug=None):
    running_loss, correct, total = [], 0, 0
    model.train()

    train_bar = tqdm(dataloader)
    for x, y in train_bar:
        x, y = x.to(device), y.to(device).long()

        if mix_aug is not None:
            x, y = mix_aug(x, y)

        optimizer.zero_grad()

        _, pred = model(x)

        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        # predicted_class = pred.argmax(dim=1, keepdim=False)

        # total += y.numel()
        # correct += (predicted_class == y).sum().item()

        running_loss.append(loss.item())
        train_bar.set_description(
            f'Epoch: [{epoch}] Loss: {round(sum(running_loss) / len(running_loss), 6)}')
    # acc = correct / total
    acc = None
    return sum(running_loss), acc


def validation_step(model, dataloader, device):
    correct, total = 0, 0
    model.eval()

    validation_bar = tqdm(dataloader)
    with torch.no_grad():
        for x, y in validation_bar:
            x, y = x.to(device), y.to(device)

            _, pred = model(x)

            predicted_class = pred.argmax(dim=1, keepdim=False)

            total += y.numel()
            correct += (predicted_class == y).sum().item()

            acc = correct / total
            validation_bar.set_description(
                f'accuracy is {round(acc*100, 2)}% until now.')
    return acc

# %%
loss_hisroty, train_acc_hist, val_acc_hist = [], [], []

since = time.time()
for epoch in range(1, EPOCHS+1):

    loss, train_acc = train_step(model, train_dataloader, optimizer, criterion, device, epoch)
    val_acc = validation_step(model, val_dataloader, device)

    loss_hisroty.append(loss)
    train_acc_hist.append(train_acc)
    val_acc_hist.append(val_acc)

print('Training Finished.')
print(f'elapsed time is {time.time() - since}')

# %%

# %%
torch.save(model, 'model1_73.pth')

# %%
# model2 = torch.load('model1_72.pth')
# model2.eval()
# model2(torch.randn(32, 3, 224, 224).to(device))[-1].shape

# %%
import configparser
config = configparser.ConfigParser()
config.read('confs/main.ini')


# %%
config['TRAIN'].getint('BATCH_SIZE')

# %%


# %%
input = Image.open('data/imagenette2-320/train/n01440764/ILSVRC2012_val_00017699.JPEG')
input = p(input)
input = input.unsqueeze(0)
input.shape

# %%
transforms.functional.to_pil_image(input[0])

# %%
input = input.permute(0, 2, 3, 1).unfold(1, PATCH_SIZE, PATCH_SIZE)\
        .unfold(2, PATCH_SIZE, PATCH_SIZE).contiguous()\
        .view(-1, 3, PATCH_SIZE, PATCH_SIZE)
input.shape

# %%
class FNN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.BatchNorm1d(hidden_features),
        )
        self.fc2 = nn.Linear(hidden_features, hidden_features)

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x) + shortcut
        return x

# %%
# for i in range(input.shape[0]):
#     display(transforms.functional.to_pil_image(input[i]))

# %%
input = input.view(-1, 3*PATCH_SIZE*PATCH_SIZE)
input.shape

# %%
feature_extractor = FNN(3*PATCH_SIZE*PATCH_SIZE)

# %%
input = feature_extractor(input)
input = input.view(BATCH_SIZE, -1, 3*PATCH_SIZE*PATCH_SIZE)
input.shape

# %%
sim = input @ input.transpose(-1, -2)
sim.shape

# %%
graph = sim.topk(5, dim=-1).indices
graph.shape

# %%
neibor_features = input[torch.arange(BATCH_SIZE).unsqueeze(-1).expand(-1, 196).unsqueeze(-1), graph]
neibor_features.shape

# %%
torch.stack([input, (neibor_features - input.unsqueeze(-2)).amax(dim=-2)], dim=-1).shape

# %%



