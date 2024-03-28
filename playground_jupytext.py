# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from model import VGNN
from dataset import ImageNetteDataset
from tqdm import tqdm
import time

# %%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
print(device)

# %%
%load_ext autoreload
%autoreload 2

# %%
PATCH_SIZE = 16
BATCH_SIZE = 16
EPOCHS = 10

# %%


# %%
class Classifier(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.backbone = VGNN()
        self.fc1 = nn.Linear(150528, 256)
        self.fc2 = nn.Linear(256, n_classes)
        
    def forward(self, x):
        features = F.relu(self.backbone(x))
        B, N, C = features.shape
        x = F.relu(self.fc1(features.view(B, -1)))
        x = self.fc2(x)
        return features, x

# %%
model = Classifier()
model(torch.randn(32, 3, 224, 224))[-1].shape

# %%
pretrained_means = [0.485, 0.456, 0.406]
pretrained_stds = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=pretrained_means,
                         std=pretrained_stds),
])
train_dataset = ImageNetteDataset('data/imagenette2-320', split='train', transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataset = ImageNetteDataset('data/imagenette2-320', split='val', transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# %%
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# %%
def train_step(model, dataloader, optimizer, criterion, device, epoch=None):
    running_loss, correct, total = [], 0, 0
    model.train()

    train_bar = tqdm(dataloader)
    for x, y in train_bar:
        x, y = x.to(device), y.to(device).long()
        
        optimizer.zero_grad()

        _, pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        predicted_class = pred.argmax(dim=1, keepdim=False)

        total += y.numel()
        correct += (predicted_class == y).sum().item()

        running_loss.append(loss.item())
        train_bar.set_description(f'Epoch: [{epoch}] Loss: {round(sum(running_loss) / len(running_loss), 6)}')
    acc = correct / total
    return sum(running_loss), acc

# def validation_step(model, dataloader, device):
#     correct, total = 0, 0
#     model.eval()

#     validation_bar = tqdm(dataloader)
#     with torch.no_grad():
#         for x, y in validation_bar:
#             x, y = x.to(device), y.to(device)

#             _, pred = model(x)

#             predicted_class = pred.argmax(dim=1, keepdim=False)

#             total += y.numel()
#             correct += (predicted_class == y).sum().item()

#             acc = correct / total
#             validation_bar.set_description(f'accuracy is {round(acc*100, 2)}% until now.')
#     return acc

# %%
loss_hisroty, train_acc_hist, val_acc_hist = [], [], []

since = time.time()
for epoch in range(1, EPOCHS+1):

    loss, train_acc = train_step(model, train_dataloader, optimizer, criterion, device, epoch)
    # val_acc = validation_step(model, val_dataloader, device)

    loss_hisroty.append(loss)
    train_acc_hist.append(train_acc)
    # val_acc_hist.append(val_acc)

print('Training Finished.')
print(f'elapsed time is {time.time() - since}')

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



