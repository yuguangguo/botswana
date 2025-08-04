import torch
from torchvision.io import read_image
from torch.utils.data import DataLoader, Dataset
import os
import pandas as pd
from torchvision import transforms, datasets

train_data = datasets.FashionMNIST(
    root = 'data',
    train = True,
    download = True,
    transform = ToTensor()
)

test_data = datasets.FashionMNIST(
    root = 'data',
    train = False,
    download = True,
    transform = ToTensor()
)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(9, 9))
rows, cols = 3, 3
for i in range(1, rows*cols+1):
    index = torch.randint(len(train_data), size=(1, )).item()
    img, label = train_data[index]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(img.squeeze(), cmap='gray')
plt.show()

class MyImageDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(image)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


train_dataloader = DataLoader(
    train_data, batch_size=64, shuffle=True, num_workers=0    
)
test_dataloader = DataLoader(
    test_data, batch_size=64, shuffle=True, num_workers=0    
)

img_features, img_labels = next(iter(train_dataloader))
print(f"{img_features.size()}")
print(f"{img_labels.dtype}")
image = img_features[0].squeeze()
label = img_labels[0]
plt.title(labels_map[label.item()])
plt.imshow(image, cmap='gray')
plt.show()
print(label)