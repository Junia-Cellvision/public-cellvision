# Classificateur de MEA

Entrée : image (PIL.Image.Image)
Sortie : type de MEA (str)

## Utilisation du modèle entraîné

Il suffit d'importer le Classifier, voir le code d'exemple dans classifier.py.

Vous pouvez spécifier le path des poids du modèle lors de l'initialisation du classifier.

## Flow d'entraînement réalisé par train.py

1. Créer le dossier data_augmented à partir du couple (../data/images/, ../data/mea_classifier/mea_class_labels.csv)
2. Entraîner le modèle
3. Stocker les poids et les résultats
4. Stocker les meilleurs poids dans "weights+results.pt" pour l'utilisation

## Entraîner le modèle

1. Lancer train.py