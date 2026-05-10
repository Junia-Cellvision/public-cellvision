# Lib imports
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from py_markdown_table.markdown_table import markdown_table

# Base imports
import numpy as np
from collections import defaultdict
from datetime import datetime

# codebase imports
from classifier import LargeImageCNN, classify_image, init_weights
from data import augment_data, get_kfold_and_full_loaders, BATCH_SIZE
from checkpoints import CHECKPOINTS_DIR, save_best_checkpoint

print(torch.__version__)
print(f"Batch size : {BATCH_SIZE}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

augment_data()

# Hyperparamètres
N_MODELS = 5          # Nombre impair de modèles dans l'ensemble
SEEDS = [42, 137, 271, 503, 919]  # Une seed par modèle
assert len(SEEDS) == N_MODELS and N_MODELS % 2 == 1

N_FOLDS = 5           # K-fold pour déterminer le meilleur nombre d'epochs
MAX_EPOCHS = 200
MAX_EPOCHS_WITHOUT_IMPROVEMENT = 20

criterion = nn.CrossEntropyLoss()

# Chargement des splits (test set fixe, folds sur train+val, full train loader)
folds, full_train_loader, test_loader, class_names = get_kfold_and_full_loaders(n_folds=N_FOLDS)

print(f"Classes : {list(class_names)}")
print(f"Taille full train : {len(full_train_loader.dataset)}")
print(f"Taille test      : {len(test_loader.dataset)}")
print(f"Nombre de folds  : {N_FOLDS}")
print(f"Nombre de modèles: {N_MODELS}")


# ─── Utilitaires d'entraînement ───────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device).long()
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def train_one_epoch(model, optimizer, loader):
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device).long()
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def find_best_epoch_via_kfold(seed):
    """
    Entraîne un modèle sur chaque fold avec la seed donnée.
    Retourne le numéro d'epoch qui minimise la val loss moyenne sur les folds,
    ainsi que les historiques train/val par fold.
    """
    # val_loss_per_epoch[e] = liste des val losses (une par fold) à l'epoch e
    val_loss_per_epoch = defaultdict(list)
    # fold_histories[fold_idx] = {"train": [...], "val": [...]}
    fold_histories = []

    for fold_idx, (train_loader, val_loader) in enumerate(folds):
        model = LargeImageCNN(list(class_names)).to(device)
        init_weights(model, seed)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        best_loss, best_epoch, no_improve = None, 0, 0
        train_losses, val_losses = [], []

        for epoch in range(1, MAX_EPOCHS + 1):
            train_loss = train_one_epoch(model, optimizer, train_loader)
            val_loss, _ = evaluate(model, val_loader)
            train_losses.append(train_loss)
            val_losses.append(val_loss)

            if best_loss is None or val_loss < best_loss:
                best_loss, best_epoch, no_improve = val_loss, epoch, 0
            else:
                no_improve += 1
                if no_improve > MAX_EPOCHS_WITHOUT_IMPROVEMENT:
                    break

        fold_histories.append({"train": train_losses, "val": val_losses})

        for e, vl in enumerate(val_losses):
            val_loss_per_epoch[e].append(vl)

        print(f"  Fold {fold_idx+1}/{N_FOLDS} → early stop epoch {best_epoch}, best_loss={best_loss:.4f}")

    # Epoch avec la val loss moyenne minimale sur les folds
    avg_losses = {e: np.mean(losses) for e, losses in val_loss_per_epoch.items()}
    best_epoch = min(avg_losses, key=avg_losses.get) + 1  # +1 car 0-indexé
    best_avg_loss = avg_losses[best_epoch - 1]
    return best_epoch, best_avg_loss, avg_losses, fold_histories


def train_on_full_data(seed, n_epochs):
    """Entraîne un modèle sur TOUTES les données train+val pendant n_epochs epochs."""
    model = LargeImageCNN(list(class_names)).to(device)
    init_weights(model, seed)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, n_epochs + 1):
        loss = train_one_epoch(model, optimizer, full_train_loader)
        if epoch % 20 == 0 or epoch == n_epochs:
            print(f"  epoch {epoch:03d}/{n_epochs} | train loss={loss:.4f}")

    return model, model.state_dict()

# Entraînement de l'ensemble
# Pour chaque seed :
#   1. K-fold → trouver le meilleur nombre d'epochs (avg val loss)
#   2. Réentraîner sur TOUTES les données pour ce nombre d'epochs
#   3. Sauvegarder le modèle final

ensemble_models = []
ensemble_weights = []
model_best_epochs = []
model_avg_losses = []
all_fold_histories = []

print(f"Démarrage entraînement ensemble ({N_MODELS} modèles, {N_FOLDS} folds chacun)\n")

for model_idx, seed in enumerate(SEEDS):
    print(f"{'='*60}")
    print(f"Modèle {model_idx+1}/{N_MODELS} | seed={seed}")
    print(f"  Étape 1 : K-fold ({N_FOLDS} folds) pour trouver le meilleur nombre d'epochs...")

    best_epoch, best_avg_loss, _, fold_histories = find_best_epoch_via_kfold(seed)
    print(f"  → Meilleur epoch : {best_epoch} (val loss moy. = {best_avg_loss:.4f})")

    print(f"  Étape 2 : Entraînement sur toutes les données ({best_epoch} epochs)...")
    model, weights = train_on_full_data(seed, best_epoch)

    ensemble_models.append(model)
    ensemble_weights.append(weights)
    model_best_epochs.append(best_epoch)
    model_avg_losses.append(best_avg_loss)
    all_fold_histories.append(fold_histories)
    print()

print(f"Ensemble entraîné.")
print(f"Epochs par modèle : {model_best_epochs}")
print(f"Val loss moy. par modèle (k-fold) : {[f'{l:.4f}' for l in model_avg_losses]}")

# Sauvegarde du checkpoint
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
checkpoint_path = CHECKPOINTS_DIR / f"ensemble_{timestamp}_results+weights.pt"

torch.save({
    "ensemble_weights": ensemble_weights,
    "classes": list(class_names),
    "seeds": SEEDS,
    "n_folds": N_FOLDS,
    "n_models": N_MODELS,
    "model_best_epochs": model_best_epochs,
    "best_loss": float(np.mean(model_avg_losses)),
    "acc_per_class": {},  # rempli dans la cellule suivante
}, checkpoint_path)

print(f"\nCheckpoint sauvegardé : {checkpoint_path}")

# Évaluation finale de l'ensemble sur le test set

@torch.no_grad()
def evaluate_ensemble_per_class(models, dataloader, device):
    for m in models:
        m.eval()

    correct = defaultdict(int)
    total = defaultdict(int)
    all_confidences = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Moyenne des probabilités sur tous les modèles
        all_probs = []
        for m in models:
            logits = m(images)
            all_probs.append(torch.softmax(logits, dim=1))
        avg_probs = torch.stack(all_probs).mean(dim=0)

        preds = avg_probs.argmax(dim=1)
        confidences = avg_probs.max(dim=1).values

        all_confidences.extend(confidences.cpu().tolist())

        for y_true, y_pred in zip(labels, preds):
            total[int(y_true)] += 1
            if y_true == y_pred:
                correct[int(y_true)] += 1

    acc_per_class = {cls: correct[cls] / total[cls] for cls in total}
    mean_confidence = float(np.mean(all_confidences))
    return acc_per_class, mean_confidence


acc_per_class, mean_confidence = evaluate_ensemble_per_class(ensemble_models, test_loader, device)

table_results = []
results = {}
for cls_id, acc in acc_per_class.items():
    results[class_names[cls_id]] = acc
    table_results.append({"Class": class_names[cls_id], "Précision": f"{acc:.4f}"})

overall_acc = sum(acc_per_class.values()) / len(acc_per_class)
table_results.append({"Class": "Toutes (macro)", "Précision": f"{overall_acc:.4f}"})

# Mise à jour du checkpoint avec les résultats
saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
saved["acc_per_class"] = {int(k): v for k, v in acc_per_class.items()}
saved["results"] = {str(k): v for k, v in results.items()}
torch.save(saved, checkpoint_path)

print(f"Loss moyenne des folds : {np.mean(model_avg_losses):.4f}")
print(f"Confiance moyenne (test) : {mean_confidence:.4f}")
print(markdown_table(table_results).set_params(
    padding_width=3,
    padding_weight='right',
    row_sep='markdown'
).get_markdown())

# Courbes d'entraînement par fold (moyenne sur les N_MODELS seeds)
# Pour chaque epoch, on moyenne sur les modèles qui ont atteint cet epoch
def _avg_across_models(fold_idx, key):
    curves = [all_fold_histories[m][fold_idx][key] for m in range(N_MODELS)]
    max_len = max(len(c) for c in curves)
    return [np.mean([c[e] for c in curves if e < len(c)]) for e in range(max_len)]

fig, axes = plt.subplots(1, N_FOLDS, figsize=(4 * N_FOLDS, 4), sharey=False)
for fold_idx, ax in enumerate(axes):
    avg_train = _avg_across_models(fold_idx, "train")
    avg_val = _avg_across_models(fold_idx, "val")
    ax.plot(avg_train, label="train loss")
    ax.plot(avg_val, label="val loss")
    ax.set_title(f"Fold {fold_idx+1}")
    ax.set_xlabel("Epoch")
    ax.legend(fontsize=7)
plt.suptitle(f"Courbes moyennées sur {N_MODELS} seeds", fontsize=10)
plt.tight_layout()
plt.savefig("learn-curve.png", dpi=300, bbox_inches="tight")
plt.close()

# Inspection des prédictions de l'ensemble sur quelques exemples du test set
num_examples = 12
examples = []
for x, y in test_loader:
    for i in range(x.size(0)):
        examples.append((x[i:i+1], y[i:i+1]))
        if len(examples) >= num_examples:
            break
    if len(examples) >= num_examples:
        break

cols = 3
rows = (num_examples + cols - 1) // cols

plt.figure(figsize=(12, rows * 4))

for i, (x, y_true) in enumerate(examples):
    x = x.to(device)

    pred_class, confidence = classify_image(x, ensemble_models)
    true_class = class_names[int(y_true)]

    print(f"Example {i} | Vrai: {true_class} | Prédit: {pred_class} | Confiance: {confidence:.2%}")

    plt.subplot(rows, cols, i + 1)
    plt.imshow(x.squeeze().cpu(), cmap="gray")
    plt.axis("off")
    correct = true_class == pred_class
    plt.title(f"{'OK' if correct else 'KO'} {pred_class}\n{confidence:.0%}", fontsize=8)

plt.tight_layout()
plt.savefig("predictions.png", dpi=300, bbox_inches="tight")
plt.close()

save_best_checkpoint()