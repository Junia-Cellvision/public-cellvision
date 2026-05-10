# Scripts d'entraînement — CellVision

Ce dossier contient les scripts d'entraînement et d'inférence pour les modèles de détection utilisés dans CellVision. Les données d'entraînement ne sont **pas** incluses.

---

## Contenu du dossier `yolo/`

| Fichier | Description |
|
| `training.ipynb` | Notebook Jupyter : entraînement du modèle |
| `training.py` | Version Python exportée depuis le notebook |
| `run_trained_model.ipynb` | Notebook Jupyter : inférence avec le modèle entraîné |
| `run_trained_model.py` | Version Python exportée depuis le notebook |

---

## Importer les `.py` dans Jupyter

*Si vous ne souhaitez pas importer les .ipynb, vous pouvez suivre les étapes suivantes :*

Les fichiers `.py` sont des exports directs de notebooks (générés via *File → Download as → Executale script*). 

Pour les réimporter en tant que notebooks Jupyter :

```bash
pip install jupytext
jupytext --to notebook training.py           # → training.ipynb
jupytext --to notebook run_trained_model.py  # → run_trained_model.ipynb
```

Ou manuellement : créer un nouveau notebook, puis copier-coller chaque bloc délimité par `# In[ ]:` dans une cellule distincte.

---

## Structure attendue du dataset

Le script d'entraînement attend un fichier `.zip` à l'emplacement suivant :

Le dataset provient d'un export *[LabelStudio](https://labelstud.io/guide/install.html)*

```
yolo/
└── dataset/
    └── some_dataset.zip
```

Le zip doit respecter la structure YOLO standard :

```
some_dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
└── labels/
    ├── image_001.txt
    ├── image_002.txt
    └── ...
```

### Format des annotations (YOLO segmentation)

Chaque fichier `.txt` dans `labels/` correspond à l'image du même nom. Chaque ligne décrit un objet :

```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```

- Coordonnées normalisées entre 0 et 1 (relatives à la taille de l'image).
- Pour les **bounding boxes** : `<class_id> <x_center> <y_center> <width> <height>`.
- Pour les **masques de segmentation** : `<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>`.

### Classes

| ID | Nom |
|---|---|
| `0` | `electrode_bar` |
| `1` | `electrode_end` |

---

## Entraînement (`training.ipynb`)

### Prérequis

```bash
pip install ultralytics matplotlib
```

Un GPU CUDA est fortement recommandé. Le script affiche automatiquement la disponibilité CUDA au démarrage.

### Configuration (cellule 3)

```python
zip_path     = './dataset/some_dataset.zip'  # chemin vers le zip
dataset_root = './custom_dataset'            # dossier d'extraction
```

### Ce que fait le script

1. **Extraction** du zip dans `./custom_dataset/`, suppression des artefacts macOS (`.DS_Store`, `__MACOSX`).
2. **Split 80/20** automatique (seed fixe à 42) entre train et validation.
3. **Génération** du fichier `data.yaml` avec les chemins et les classes.
4. **Sauvegarde préventive** de tout modèle `.pt` existant dans `./model_backup/`.
5. **Entraînement** du modèle `yolo11m-seg` (segmentation) pendant 200 époques.
6. **Promotion** du meilleur modèle (`best.pt`) vers `../data/models/`.
7. **Validation** sur le jeu de validation, avec génération de graphiques.
8. **Inférence** sur les images placées dans `./images/raw/`.

### Paramètres d'entraînement

| Paramètre | Valeur |
|---|---|
| Modèle de base | `yolo11m-seg.pt` |
| Époques | 200 |
| Taille d'image | 640 × 640 |
| Batch size | 4 |
| Early stopping | patience = 30 |
| Augmentation | rotation ±45°, mosaic activé |

### Résultats

- Modèle final : `../data/models/best.pt`
- Graphiques de validation : `../data/models/validation/`
- Résultats d'inférence de test : `../data/models/inference/`

---

## Inférence (`run_trained_model.ipynb`)

### Prérequis

```bash
pip install ultralytics opencv-python matplotlib
```

### Structure attendue

```
yolo/
├── model_result/
│   └── best.pt          # modèle entraîné
└── images/
    └── raw/             # images à analyser (.jpg, .jpeg, .png)
```

### Ce que fait le script

- Charge le modèle depuis `./model_result/best.pt`.
- Prédit sur toutes les images dans `./images/raw/`.
- Sauvegarde pour chaque image :
  - Une **image annotée** dans `./model_result/inference/images/`
  - Un **fichier JSON** dans `./model_result/inference/labels/`

### Format de sortie JSON

```json
{
    "filename": "image_001.jpg",
    "electrode_bars_poly": [
        [[x1, y1], [x2, y2], ...]
    ],
    "electrode_ends_box": [
        [x_min, y_min, x_max, y_max]
    ]
}
```

- `electrode_bars_poly` : polygones de segmentation (pixels absolus).
- `electrode_ends_box` : bounding boxes au format `xyxy` (pixels absolus).

## Contenu du dossier `nnUnet/`

| Fichier | Description |
|---|---|
| `train.py` | Script Python pour préparer l'environnement nnU-Net, lancer l'entraînement et exporter le modèle. |
| `Models/` | Dossier de sortie pour les modèles exportés au format `.zip`. |

---

## Entraînement nnU-Net (`nnUnet/train.py`)

### Prérequis

Utiliser l'environement conda/mamba (fichier env.yml) afin d'avoir toutes les dépendances nécéssaires.

Le script configure automatiquement les variables d'environnement nécessaires à nnU-Net :

- `nnUNet_raw`
- `nnUNet_preprocessed`
- `nnUNet_results`

Par défaut, elles pointent vers un dossier local `nnUnet/NNunet_data` à côté de `train.py`. Vous pouvez surcharger ce chemin avec l'argument `--data-root`.

Le script crée automatiquement les trois dossiers nécessaires si ils n'existent pas.

### Ce que fait le script

1. Initialise l'environnement nnU-Net et crée les dossiers attendus.
2. Vérifie et prépare le dataset avec `nnUNetv2_plan_and_preprocess`.
3. Copie le plan personnalisé si nécessaire.
4. Lance le prétraitement avec `nnUNetv2_preprocess`.
5. Entraîne le modèle sur les 5 folds avec `nnUNetv2_train`.
6. Recherche la meilleure configuration via `nnUNetv2_find_best_configuration`.
7. Exporte le modèle final dans `./Models/NNunet_mock_model.zip`.

### Configuration principale

Dans le bloc `__main__` de [nnUnet/train.py](nnUnet/train.py), les paramètres principaux sont :

| Paramètre       | Valeur                           |
| -----------------| ----------------------------------|
| ID dataset      | `X`                              |
| Trainer         | `nnUNetTrainer_250epochs`        |
| Plan            | `nnUNetPlans`                    |
| Nombre de folds | `5`                              |
| Export          | `./Models/NNunet_mock_model.zip` |

### Résultats

- Modèle exporté : `nnUnet/Models/NNunet_mock_model.zip`
- Journal d'exécution : `nnUnet/training_debug.log`

### Remarques

- Le script est conçu pour un dataset déjà préparé au format attendu par nnU-Net.
- Le nombre d'époques est fixé à 250 via la classe `nnUNetTrainer_250epochs`.
- L'entraînement utilise CUDA si disponible via PyTorch.

## Créer un dataset nnU-Net

Si vous partez de vos images et masques de segmentation, voici la structure attendue par nnU-Net pour un dataset de type 2D.

### Arborescence

Le dossier doit être créé dans `nnUNet_raw/` avec un nom du type `Dataset00X_CellVision` : (ou x est le numéro du dataset)

```text
nnUNet_raw/
└── Dataset00X_CellVision/
    ├── imagesTr/
    ├── labelsTr/
    └── dataset.json
```

### Règles de nommage

- Les images d'entraînement sont placées dans `imagesTr/`.
- Les masques de segmentation correspondants sont placés dans `labelsTr/`.
- Chaque image et son masque doivent partager le même identifiant de base.
- Utilisez un format homogène pour tout le dataset et gardez la même convention de nommage entre images et masques.

Exemple :

```text
imagesTr/cellvision_001_0000.png
labelsTr/cellvision_001.png
imagesTr/cellvision_002_0000.png
labelsTr/cellvision_002.png
```

### Métadonnées `dataset.json`

Le fichier `dataset.json` décrit le dataset, les modalités et les classes. Un exemple minimal :

```json
{
    "channel_names": {
        "0": "RGB"
    },
    "labels": {
        "background": 0,
        "electrode_bar": 1,
        "electrode_end": 2
    },
    "numTraining": 2,
    "file_ending": ".png"
}
```

### Étapes de préparation

1. Convertir les images en un format homogène.
2. Générer les masques de segmentation avec les mêmes dimensions que les images.
3. Placer les paires image/masque dans `imagesTr/` et `labelsTr/`.
4. Vérifier que `dataset.json` correspond bien aux classes réelles.
5. Lancer [nnUnet/train.py](nnUnet/train.py) pour vérifier le prétraitement et l'entraînement.

### Point d'attention

- Le script actuel utilise `Dataset003_CellVision` dans `__main__`, donc votre dataset doit conserver cet identifiant ou vous devez adapter `DATASET_ID` dans [nnUnet/train.py](nnUnet/train.py).
- Si vous changez les classes ou le nombre de canaux, mettez aussi à jour `dataset.json` et le plan utilisé par nnU-Net.

