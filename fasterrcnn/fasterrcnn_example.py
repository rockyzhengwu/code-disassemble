import torch
from fasterrcnn.faster_rcnn import fasterrcnn_resnet50_fpn
import torchvision
import dataset

model = fasterrcnn_resnet50_fpn(pretrained=True)
images, targets = dataset.load_data()
to_tensor = torchvision.transforms.ToTensor()
images = [to_tensor(image) for image in images]
targets = [{'boxes':item['boxes'], 'labels': item['labels']} for item in targets]

# model.eval()
# For training
output = model(images, targets )
# print(output)
# optionally, if you want to export the model to ONNX: