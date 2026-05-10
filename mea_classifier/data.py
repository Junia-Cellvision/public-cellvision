# Lib imports
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
from PIL import Image

# Base imports
import numpy as np
import os
from pathlib import Path
import shutil

DATA_DIR = Path(__file__).parent / "../data"
AUGMENTED_DATA_DIR = Path(__file__).parent / "data_augmented"

# Augmentation online pour l'entraînement (appliquée à la volée dans le DataLoader)
# Complète l'augmentation offline (rotations) avec d'autres variations
# IMAGE_SIZE est importé depuis classifier pour garder une taille cohérente train/inférence
from classifier import transform, IMAGE_SIZE

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

BATCH_SIZE = 16  # Multiple de 2 pour éviter les sauts de loss
NUM_WORKERS = 4  # Workers pour le chargement parallèle des images


def augment_data():
    images_dir = DATA_DIR / "images"
    csv_file = DATA_DIR / "mea_classifier/mea_class_labels.csv"

    if AUGMENTED_DATA_DIR.exists():
        shutil.rmtree(AUGMENTED_DATA_DIR)
    os.makedirs(AUGMENTED_DATA_DIR, exist_ok=True)

    print(f"Dossier réinitialisé : {AUGMENTED_DATA_DIR}")

    df = pd.read_csv(csv_file)

    ROTATION_INTERVAL = 20
    rotations = np.arange(ROTATION_INTERVAL, 360, ROTATION_INTERVAL)

    new_data = []

    print(f"Augmentation en cours...")
    for _, row in df.iterrows():
        img_filename = row['image']
        try:
            img_path = images_dir / img_filename
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Erreur sur l'image {img_filename}: {e}")
            continue

        grayscale = transforms.Grayscale(num_output_channels=1)
        grayscale_img = grayscale(img)
        grayscale_img.save(AUGMENTED_DATA_DIR / img_filename)
        new_data.append({"class": row['class'], "image": img_filename})

        for rotation in rotations:
            rotate = transforms.RandomRotation((rotation, rotation))
            aug_img = rotate(grayscale_img)

            img_name = Path(img_filename).stem
            img_ext = Path(img_filename).suffix

            aug_filename = f"{img_name}_{rotation:03}deg{img_ext}"
            aug_img.save(AUGMENTED_DATA_DIR / aug_filename)

            new_data.append({"class": row['class'], "image": aug_filename})
        print(f"{img_filename} augmentée")

    new_df = pd.DataFrame(new_data)
    new_df.to_csv(AUGMENTED_DATA_DIR / "labels.csv", index=False)

    print(f"\nTraitement terminé.")
    print(f"Total images : {len(new_df)}")


class ImageDataset(Dataset):
    class_names: list[str]

    def __init__(self, transform=None):
        self.df = pd.read_csv(AUGMENTED_DATA_DIR / "labels.csv")
        self.transform = transform

        self.labels = pd.Categorical(self.df['class']).codes
        self.class_names = pd.Categorical(self.df['class']).categories

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_filename = self.df.iloc[idx]['image']
        img_path = AUGMENTED_DATA_DIR / img_filename

        image = Image.open(img_path).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_kfold_and_full_loaders(n_folds: int = 5, batch_size: int = BATCH_SIZE, test_size: float = 0.1, seed: int = 42):
    """
    Sépare d'abord un test set fixe (test_size %), puis applique StratifiedKFold
    sur le reste (train+val).

    Retourne :
        folds        : liste de (train_loader, val_loader) pour chaque fold
        full_loader  : DataLoader sur toutes les données non-test (pour l'entraînement final)
        test_loader  : DataLoader sur le test set
        class_names  : noms des classes
    """
    dataset_aug = ImageDataset(transform=train_transform)
    dataset_val = ImageDataset(transform=transform)

    labels = np.array(dataset_aug.labels)
    indices = np.arange(len(dataset_aug))

    # 1. Isoler le test set (identique pour tous les modèles)
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=labels,
        random_state=seed,
    )

    test_ds = Subset(dataset_val, test_idx.tolist())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 2. Full train loader (toutes les données non-test, avec augmentation)
    full_train_ds = Subset(dataset_aug, trainval_idx.tolist())
    full_loader = DataLoader(full_train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # 3. K-fold sur les données non-test
    trainval_labels = labels[trainval_idx]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for fold_train_idx, fold_val_idx in skf.split(trainval_idx, trainval_labels):
        # fold_train_idx / fold_val_idx sont des indices DANS trainval_idx
        real_train_idx = trainval_idx[fold_train_idx].tolist()
        real_val_idx = trainval_idx[fold_val_idx].tolist()

        train_ds = Subset(dataset_aug, real_train_idx)
        val_ds = Subset(dataset_val, real_val_idx)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

        folds.append((train_loader, val_loader))

    return folds, full_loader, test_loader, dataset_aug.class_names
