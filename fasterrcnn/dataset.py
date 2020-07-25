#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author: wu.zheng midday.me


import os
import torch
import random
from PIL import Image, ImageDraw

BATCH_SIZE = 6
ROOT_DIR = './data/'

def load_data():
  images = []
  targets = []
  for i in range(BATCH_SIZE):
    image_path = os.path.join(ROOT_DIR, "image_%d.png"%(i))
    label_path = os.path.join(ROOT_DIR, "image_%d.pt"%(i))
    target = torch.load(label_path)
    image = Image.open(image_path)
    images.append(image)
    targets.append(target)
  return images, targets

def get_color():
  color = tuple(random.randint(0,255) for _ in range(3))
  return color

def vis_data(image, target, show=False):
  draw =ImageDraw.Draw(image)
  boxes = target['boxes'].numpy()
  labels = target['labels'].numpy()
  for i, box in enumerate(boxes):
    box = tuple(box)
    color = get_color()
    draw.rectangle(box, outline=color) 
    draw.text((box[0]+5,box[1]+5),str(labels[i]), color)
  if show:
    image.show()
  else:
    image.save("vis_result.png")

if __name__ == "__main__":
  images, targets = load_data()
  vis_data(images[0], targets[0], show=True)
