import os
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import torchvision.transforms as T
from PIL import Image
import json
import matplotlib.pyplot as plt

class UAVVasteDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, ann_file, transforms=None):
        self.img_dir = img_dir
        self.transforms = transforms

        # Load annotations from JSON
        with open(ann_file, 'r') as f:
            data = json.load(f)  # Load full JSON structure
            self.annotations = data["annotations"]  # Access the 'annotations' part of the JSON

        # Create an index of annotations by image_id
        self.image_ids = list({ann['image_id'] for ann in self.annotations})
        self.annotations_per_image = {image_id: [] for image_id in self.image_ids}
        
        for ann in self.annotations:
            self.annotations_per_image[ann['image_id']].append(ann)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]

        # Load image
        img_filename = f"BATCH_d06_img_{image_id}.jpg"  # Assuming image filenames are structured this way
        img_path = os.path.join(self.img_dir, img_filename)
        img = Image.open(img_path).convert("RGB")

        # Get annotations for the current image
        anns = self.annotations_per_image[image_id]
        
        # Prepare the target dictionary
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        for ann in anns:
            boxes.append(ann['bbox'])  # [x, y, width, height]
            labels.append(ann['category_id'])  # Class label
            areas.append(ann['area'])
            iscrowd.append(ann['iscrowd'])

        # Convert everything to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([image_id]),
            "area": areas,
            "iscrowd": iscrowd
        }

        # Apply transformations
        if self.transforms:
            img = self.transforms(img)

        return img, target

# Define data transformations
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())  # Correct way to apply ToTensor()
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))  # Augmentation only for training
    return T.Compose(transforms)

# Initialize the model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Adjust the final classifier for the number of waste categories
num_classes = 2  # Waste (1) and background (0)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Define the paths to the images and annotation files
image_dir = "./UAVVaste/images"
annotation_file = "./UAVVaste/annotations/annotations.json"

# Load training and validation data
dataset = UAVVasteDataset(image_dir, annotation_file, get_transform(train=True))
dataset_test = UAVVasteDataset(image_dir, annotation_file, get_transform(train=False))

# Data loaders
data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))
data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=4, shuffle=False, num_workers=4, collate_fn=lambda x: tuple(zip(*x)))

# Set training parameters
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Store losses for plotting
train_losses = []
val_losses = []

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    for images, targets in data_loader:
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        train_loss += losses.item()

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    train_losses.append(train_loss / len(data_loader))
    
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, targets in data_loader_test:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
    
    val_losses.append(val_loss / len(data_loader_test))
    
    print(f"Epoch {epoch+1}, Train Loss: {train_losses[-1]}, Validation Loss: {val_losses[-1]}")

# Save the trained model for future deployment
torch.save(model.state_dict(), "uavvaste_waste_detection_model.pth")

# Plot loss curves
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
