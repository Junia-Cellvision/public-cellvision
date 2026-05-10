# Lib imports
import torch

# Base imports
from pathlib import Path


CHECKPOINTS_DIR = Path(__file__).parent.parent / "data/mea_classifier/checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)
BEST_CHECKPOINT = CHECKPOINTS_DIR / 'best_results+weights.pt'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _score(checkpoint: dict) -> float:
    """Score composite : somme des précisions par classe (plus élevé = meilleur)."""
    return sum(checkpoint.get("acc_per_class", {}).values())


def get_best():
    best_score = None
    best_checkpoint = None

    if BEST_CHECKPOINT.exists():
        best_checkpoint = torch.load(BEST_CHECKPOINT, weights_only=False, map_location=DEVICE)

    # Accepte les deux formats : simple et ensemble
    patterns = ["*_results+weights.pt", "ensemble_*_results+weights.pt"]
    paths = set()
    for pat in patterns:
        paths.update(CHECKPOINTS_DIR.glob(pat))

    for checkpoint_path in paths:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=DEVICE)
        print(checkpoint_path, checkpoint.get("best_loss", "?"))

        score = _score(checkpoint)

        if (
            best_checkpoint is None
            or best_score is None
            or score > best_score
            or (score == best_score and checkpoint.get("best_loss", float("inf")) < best_checkpoint.get("best_loss", float("inf")))
        ):
            best_score = score
            best_checkpoint = checkpoint

    if not best_checkpoint:
        raise ValueError("Aucun checkpoint trouvé")

    return best_checkpoint


def save_best_checkpoint():
    checkpoint = get_best()
    torch.save(checkpoint, BEST_CHECKPOINT)
