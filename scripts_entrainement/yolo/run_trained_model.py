#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import torch
import gc
import json
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
import glob

# Memory Cleanup
torch.cuda.empty_cache()
gc.collect()

# CONFIG
MODEL_DIR   = './model_result'
source_path = './images/raw/'

# Load model directly from model_result/
model_path = os.path.join(MODEL_DIR, 'best.pt')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"No model found at {model_path}. Run training first.")
print(f"Loading model from: {model_path}")
model = YOLO(model_path)

if not os.path.exists(source_path) or not os.listdir(source_path):
    raise ValueError(f"No images found in {source_path}")

# INFERENCE
results_generator = model.predict(
    source=source_path,
    imgsz=640,
    conf=0.25,
    save=False,
    stream=True,
    device=0,
    retina_masks=False,
    classes=[1]
)

# Save outputs inside model_result/ to keep everything in one place
images_save_dir = os.path.join(MODEL_DIR, 'inference', 'images')
labels_save_dir = os.path.join(MODEL_DIR, 'inference', 'labels')
os.makedirs(images_save_dir, exist_ok=True)
os.makedirs(labels_save_dir, exist_ok=True)

print(f"\n--- Processing & Saving Data ---")
for i, r in enumerate(results_generator):
    r = r.cpu()

    filename  = os.path.basename(r.path)
    file_stem = os.path.splitext(filename)[0]

    image_data = {
        "filename":            filename,
        "electrode_bars_poly": [],
        "electrode_ends_box":  []
    }

    if r.boxes:
        for idx, cls_tensor in enumerate(r.boxes.cls):
            cls_id     = int(cls_tensor.item())
            class_name = r.names[cls_id]

            if class_name == 'electrode_bar' or cls_id == 0:
                if r.masks is not None and len(r.masks) > idx:
                    poly = r.masks.xy[idx]
                    image_data["electrode_bars_poly"].append(poly.tolist())

            elif class_name == 'electrode_end' or cls_id == 1:
                box = r.boxes.xyxy[idx].tolist()
                image_data["electrode_ends_box"].append(box)

    # Save JSON
    json_path = os.path.join(labels_save_dir, f"{file_stem}.json")
    with open(json_path, 'w') as f:
        json.dump(image_data, f, indent=4)

    # Save annotated image
    im_bgr = r.plot(line_width=2, font_size=1.2)
    cv2.imwrite(os.path.join(images_save_dir, filename), im_bgr)

    # Display first 3
    if i < 3:
        print(f"Processed: {filename}")
        print(f"   Saved {len(image_data['electrode_bars_poly'])} bars (polygons)")
        print(f"   Saved {len(image_data['electrode_ends_box'])} ends (boxes)")
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10, 6))
        plt.imshow(im_rgb)
        plt.axis('off')
        plt.title(f"{filename} (Bar=Mask, End=Box)", fontsize=10)
        plt.show()

print(f"Done.\nImages saved to: {images_save_dir}\nData saved to:   {labels_save_dir}")


# In[ ]:




