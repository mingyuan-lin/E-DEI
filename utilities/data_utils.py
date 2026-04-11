import os
import cv2
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm

import torch
import torch.nn.functional as F

def ensure_dir(s):
    if not os.path.exists(s):
        os.makedirs(s)

def show_img(img):
    img = img.cpu().detach().numpy()
    img = np.transpose(img, (0,2,3,1))
    img = np.squeeze(img)
    img = img.clip(0,1) * 255.
    #img = cv2.cvtColor(img.clip(0,1) * 255., cv2.COLOR_BGR2GRAY)
    return img

def event_plot_cuda(image, event):
    event_image = torch.zeros_like(image)
    event = torch.sum(event.clone(), 1)
    for i in range(event_image.shape[2]):
        for j in range(event_image.shape[3]):
            if event[..., i, j] > 0:
                event_image[:, 0, i, j] = 0
                event_image[:, 1, i, j] = 0
                event_image[:, 2, i, j] = 255
            elif event[..., i, j] < 0:
                event_image[:, 0, i, j] = 255
                event_image[:, 1, i, j] = 0
                event_image[:, 2, i, j] = 0
    return event_image
