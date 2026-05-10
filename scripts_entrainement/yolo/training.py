#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install ultralytics')
get_ipython().system('pip install matplotlib')


# In[ ]:


from IPython import display
import ultralytics
from ultralytics import YOLO
import torch
import os
import shutil
import zipfile
import glob
import yaml
import random
import gc

display.clear_output()
ultralytics.checks()
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")


# In[ ]:


# Configuration
zip_path = './dataset/some_dataset.zip'
dataset_root = os.path.abspath('./custom_dataset')
train_txt_path = os.path.join(dataset_root, 'train.txt')
val_txt_path = os.path.join(dataset_root, 'val.txt')
yaml_path = os.path.join(dataset_root, 'data.yaml')

# Reset & Extract
if os.path.exists(dataset_root):
    shutil.rmtree(dataset_root)
os.makedirs(dataset_root)

if not os.path.exists(zip_path):
    raise FileNotFoundError(f"STOP: '{zip_path}' not found. Please upload it.")

print(f"Unzipping {zip_path}...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_root)

# Scan and Clean (Remove MACOSX and collect images)
all_images = []
print("Scanning and cleaning dataset...")

for root, dirs, files in os.walk(dataset_root, topdown=False):
    if "__MACOSX" in dirs:
        shutil.rmtree(os.path.join(root, "__MACOSX"))
        dirs.remove("__MACOSX")

    for f in files:
        if f.startswith("._") or f.endswith(".cache") or f == ".DS_Store":
            os.remove(os.path.join(root, f))
            continue

        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            all_images.append(os.path.join(root, f))

if not all_images:
    raise ValueError("ERROR: No images found in the zip file.")

# Split Data (80/20)
random.seed(42)
random.shuffle(all_images)
split_idx = int(len(all_images) * 0.8)
train_imgs, val_imgs = all_images[:split_idx], all_images[split_idx:]

# Write Paths to .txt
with open(train_txt_path, 'w') as f:
    f.write('\n'.join(train_imgs))
with open(val_txt_path, 'w') as f:
    f.write('\n'.join(val_imgs))

yaml_content = {
    'path': dataset_root,
    'train': train_txt_path,
    'val': val_txt_path,
    'names': {0: 'electrode_bar', 1: 'electrode_end'}
}

with open(yaml_path, 'w') as f:
    yaml.dump(yaml_content, f, sort_keys=False)

print(f"\nSUCCESS! Dataset ready.")
print(f"Total: {len(all_images)} | Train: {len(train_imgs)} | Val: {len(val_imgs)}")
print(f"Config saved at: {yaml_path}")


# In[ ]:


MODEL_DIR    = '../data/models' # clean output folder
BACKUP_DIR   = './model_backup' # old models go here
TRAIN_NAME   = 'electrode_YOLO_HighRes'
BEST_PT_NAME = 'best.pt'

def backup_existing_models(model_dir, backup_dir):
    """Move any existing .pt files to backup before new training."""
    os.makedirs(backup_dir, exist_ok=True)
    existing = glob.glob(os.path.join(model_dir, '**', '*.pt'), recursive=True)
    if not existing:
        print("No existing models to backup.")
        return
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_backup = os.path.join(backup_dir, f'backup_{timestamp}')
    os.makedirs(run_backup, exist_ok=True)
    for pt in existing:
        dest = os.path.join(run_backup, os.path.basename(pt))
        shutil.move(pt, dest)
        print(f"Backed up: {pt} → {dest}")

def clean_previous_run(runs_root, train_name):
    """Delete any existing YOLO run folders with this name before training."""
    pattern = os.path.join(runs_root, '**', train_name + '*')
    matches = glob.glob(pattern, recursive=True)
    for folder in matches:
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            print(f"Removed old run folder: {folder}")

def find_and_promote_best_model(runs_root, train_name, model_dir):
    """
    Automatically finds where YOLO actually saved the run,
    regardless of how it nested the folders.
    """
    # Search recursively for the run folder
    pattern = os.path.join(runs_root, '**', train_name + '*', 'weights', 'best.pt')
    matches = glob.glob(pattern, recursive=True)

    if not matches:
        # Print what actually exists to help debug
        print("Could not find best.pt. Searching for any best.pt under runs/...")
        all_pts = glob.glob(os.path.join(runs_root, '**', 'best.pt'), recursive=True)
        if all_pts:
            print(f"Found these instead:\n" + "\n".join(all_pts))
            best_src = all_pts[0]  # take the most recent one
        else:
            raise FileNotFoundError(f"No best.pt found anywhere under {runs_root}")
    else:
        # Sort by modification time, take the most recent
        best_src = sorted(matches, key=os.path.getmtime)[-1]

    print(f"Found model at: {best_src}")
    os.makedirs(model_dir, exist_ok=True)
    best_dst = os.path.join(model_dir, 'best.pt')
    shutil.copy2(best_src, best_dst)
    print(f"Saved to: {best_dst}")

    # Clean up the entire run folder
    run_folder = best_src.replace(os.sep + 'weights' + os.sep + 'best.pt', '')
    shutil.rmtree(run_folder, ignore_errors=True)
    print(f"Cleaned up: {run_folder}")
    return best_dst


# In[ ]:


torch.cuda.empty_cache()
gc.collect()

RUNS_ROOT = './runs'

backup_existing_models(MODEL_DIR, BACKUP_DIR)
clean_previous_run(RUNS_ROOT, TRAIN_NAME)

model = YOLO('yolo11m-seg.pt')
results = model.train(
    data=yaml_path,
    epochs=200,
    imgsz=640,
    batch=4,
    device=0 if torch.cuda.is_available() else 'cpu',
    project=RUNS_ROOT,
    name=TRAIN_NAME,
    mask_ratio=1,
    overlap_mask=True,
    degrees=45,
    mosaic=1.0,
    patience=30,
    close_mosaic=20,
    amp=True,
    optimizer='auto',
    val=False
)

BEST_MODEL_PATH = find_and_promote_best_model(RUNS_ROOT, TRAIN_NAME, MODEL_DIR)
print(f"Model ready at: {BEST_MODEL_PATH}")


# In[ ]:


torch.cuda.empty_cache()
gc.collect()

model = YOLO(BEST_MODEL_PATH)
metrics = model.val(
    data=yaml_path,
    split='val',
    imgsz=640,
    batch=1,
    device=0,
    plots=True,
    project=MODEL_DIR,      # val plots also land in model_result/
    name='validation'
)
print(f"\nFinal Box mAP50-95: {metrics.box.map}")
print(f"Final Mask mAP50-95: {metrics.seg.map}")


# In[ ]:


torch.cuda.empty_cache()

target_folder = './images/raw/'
extensions    = ('*.jpg', '*.jpeg', '*.png', '*.JPG')
test_images   = []
for ext in extensions:
    test_images.extend(glob.glob(os.path.join(target_folder, ext)))
if not test_images:
    raise ValueError(f"No images found in {target_folder}")

print(f"Loading model: {BEST_MODEL_PATH}")
model   = YOLO(BEST_MODEL_PATH)
results = model.predict(
    source=test_images,
    save=True,
    conf=0.25,
    project=MODEL_DIR,
    name='inference'
)

print("\n--- Results ---")
for result in results:
    save_dir   = result.save_dir
    file_name  = os.path.basename(result.path)
    final_path = os.path.join(save_dir, file_name)
    print(f"Detected: {file_name}")
    if os.path.exists(final_path):
        display(Image(filename=final_path, width=600))
    print("-" * 20)


# In[ ]:




