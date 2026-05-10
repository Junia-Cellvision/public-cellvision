# Lib imports
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Base imports
from pathlib import Path

from mea_classifier.checkpoints import BEST_CHECKPOINT


class LargeImageCNN(nn.Module):
    """
    CNN adapté aux images très haute résolution (>=1024x1024)
    Approche 1 : downsampling progressif + global pooling
    """

    output_classes: list[str]

    def __init__(self, output_classes: list[str]):
        super().__init__()

        self.output_classes = output_classes

        self.dropout = nn.Dropout(p=0.2)

        self.features = nn.Sequential(
            # 1024 -> 512
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 512 -> 256
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 256 -> 128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 128 -> 64
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # 64 -> 32
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),

            # Taille indépendante de la résolution d'entrée
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, len(output_classes))
        )

    def forward(self, x):
        # x: [B, 1, H, W] avec H,W >= 1024
        x = self.features(x)      # [B, 512, 1, 1]
        x = x.flatten(1)          # [B, 512]
        x = self.classifier(x)    # [B, n_classes]
        return x


def init_weights(model: nn.Module, seed: int):
    """
    Initialise les poids du modèle avec une seed spécifique.
    Permet de varier le point de départ de l'entraînement pour l'ensemble.
    """
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


IMAGE_SIZE = 512  # Taille commune pour toutes les images (résolutions variables dans le dataset)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

@torch.no_grad()
def classify_image(
    image: torch.Tensor,
    models: list[LargeImageCNN],
)-> tuple[str, float]:
    """
    Retourne (class, confidence)
    - class : Nom de la classe détectée
    - confidence : Score de confiance
    """

    classes: list[str] = models[0].output_classes

    probs_list = []

    # 1) Récupérer toutes les distributions de probabilité
    for i, model in enumerate(models):
        logits = model(image)
        probs = torch.softmax(logits, dim=1)  # shape: [1, C]
        probs_list.append(probs.squeeze(0))   # shape: [C]

    probs_stack = torch.stack(probs_list)     # shape: [N, C]

    # 2) Moyenne des probabilités
    mean_probs = probs_stack.mean(dim=0)      # shape: [C]

    # 3) Entropie de la moyenne
    entropy_mean = -torch.sum(mean_probs * torch.log(mean_probs + 1e-12))

    # 4) Entropie moyenne des modèles
    entropy_models = -torch.sum(
        probs_stack * torch.log(probs_stack + 1e-12),
        dim=1
    )  # shape: [N]

    mean_entropy = entropy_models.mean()

    # 5) Mutual Information
    mutual_information = entropy_mean - mean_entropy

    # 6) Prédiction finale (moyenne des probs)
    y_pred = int(torch.argmax(mean_probs).item())
    pred_class = classes[y_pred]

    # 7) Score de confiance (option simple)
    confidence = mean_probs[y_pred].item()

    final_score = confidence * torch.exp(-mutual_information)

    return pred_class, final_score.item()


class MEAClassifier():
    models: list[LargeImageCNN]

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not checkpoint_path:
            checkpoint_path = BEST_CHECKPOINT

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if "ensemble_weights" in checkpoint:
            # Checkpoint ensemble : plusieurs jeux de poids
            self.models = []
            for weights in checkpoint["ensemble_weights"]:
                model = LargeImageCNN(checkpoint["classes"])
                model.load_state_dict(weights)
                model.to(self.device)
                model.eval()
                self.models.append(model)
        else:
            # Checkpoint simple (rétrocompatibilité)
            model = LargeImageCNN(checkpoint["classes"])
            model.load_state_dict(checkpoint["weights"])
            model.to(self.device)
            model.eval()
            self.models = [model]

    @torch.no_grad()
    def classify(self, img_data: Image.Image) -> tuple[str, float]:
        """
        img_data: PIL.Image
        retourne: (nom de la classe (str), confiance (float))
        """
        image = img_data.convert("RGB")
        x = transform(image).unsqueeze(0).to(self.device)
        return classify_image(x, self.models)
