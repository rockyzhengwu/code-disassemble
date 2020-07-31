
import torch
import cv2
import numpy as np

def vis_box(image_tensor, boxes, out_name=None):
  image = image_tensor
  image_mean = [0.485, 0.456, 0.406]
  image_std = [0.229, 0.224, 0.225]
  mean = torch.as_tensor(image_tensor)
  std = torch.as_tensor(image_std)
  image = image * std[:,None, None] + mean[:,None, None]
  image = image_tensor.permute((1,2,0))
  image = image * 255.0
  image = image.cpu().numpy()
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  for box in boxes:
    box = box.cpu().numpy()
    box = np.int0(box)
    box = [int(i) for i in box]
    p1 = (box[0], box[1])
    p2 = (box[2], box[3])
    cv2.rectangle(image, p1, p2,(0,255,0), 2)
  if out_name is None:
    cv2.imwrite("vis_box.png",image)
  else:
    cv2.imwrite(out_name, image)
