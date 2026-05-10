import json
import logging
import os
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_DATA_ROOT = Path(os.environ.get("NNUNET_DATA_ROOT", SCRIPT_DIR / "NNunet_data"))
DEFAULT_DATASET_ID = 3
DEFAULT_REFERENCE_DATASET_ID = 2
DEFAULT_DATASET_SUFFIX = "CellVision"
DEFAULT_CHANNEL_NAMES = {"0": "RGB"}
DEFAULT_LABELS = {"background": 0, "electrode_bar": 1, "electrode_end": 2}
DEFAULT_OUTPUT = SCRIPT_DIR / "Models" / "NNunet_mock_model.zip"


def setup_environment(path=None):
	base_path = Path(path) if path is not None else DEFAULT_DATA_ROOT
	base_path = base_path.expanduser().resolve()

	os.environ["nnUNet_raw"] = str(base_path / "nnUNet_raw")
	os.environ["nnUNet_preprocessed"] = str(base_path / "nnUNet_preprocessed")
	os.environ["nnUNet_results"] = str(base_path / "nnUNet_results")

	pythonpath = os.environ.get("PYTHONPATH", "")
	os.environ["PYTHONPATH"] = str(PROJECT_ROOT) if not pythonpath else f"{PROJECT_ROOT}:{pythonpath}"

	for key in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
		Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


setup_environment()

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(levelname)s - %(message)s",
	handlers=[
		logging.FileHandler(SCRIPT_DIR / "training_debug.log"),
		logging.StreamHandler(sys.stdout),
	],
)


class nnUNetTrainer_250epochs(nnUNetTrainer):
	def init(
		self,
		plans: dict,
		configuration: str,
		fold: int,
		dataset_json: dict,
		device: torch.device = torch.device("cuda"),
	):
		super().init(plans, configuration, fold, dataset_json, device)
		self.num_epochs = 250


def run_command(command):
	try:
		logging.info("Exécution : %s", " ".join(command))
		subprocess.run(command, check=True, env=os.environ)
	except subprocess.CalledProcessError as e:
		logging.error("Erreur : %s", e)
		raise


def dataset_name(dataset_id: int) -> str:
	return f"Dataset{dataset_id:03d}_{DEFAULT_DATASET_SUFFIX}"


def ensure_dataset_scaffold(dataset_id: int):
	raw_root = Path(os.environ["nnUNet_raw"])
	dataset_dir = raw_root / dataset_name(dataset_id)
	dataset_dir.mkdir(parents=True, exist_ok=True)

	for subdir in ["imagesTr", "labelsTr", "imagesTs"]:
		(dataset_dir / subdir).mkdir(parents=True, exist_ok=True)

	dataset_json = dataset_dir / "dataset.json"
	if not dataset_json.exists():
		template = {
			"channel_names": DEFAULT_CHANNEL_NAMES,
			"labels": DEFAULT_LABELS,
			"numTraining": 0,
			"file_ending": ".png",
		}
		dataset_json.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
		logging.info("Création du template dataset.json : %s", dataset_json)

	return dataset_dir


def validate_dataset(dataset_dir: Path):
	images_tr = dataset_dir / "imagesTr"
	labels_tr = dataset_dir / "labelsTr"
	dataset_json = dataset_dir / "dataset.json"

	if not dataset_json.exists():
		raise FileNotFoundError(f"dataset.json manquant dans {dataset_dir}")

	image_files = sorted(path for path in images_tr.glob("*") if path.is_file())
	label_files = sorted(path for path in labels_tr.glob("*") if path.is_file())

	if not image_files:
		raise FileNotFoundError(f"Aucune image trouvée dans {images_tr}. Placez vos images avant de lancer l'entraînement.")
	if not label_files:
		raise FileNotFoundError(f"Aucun masque trouvé dans {labels_tr}. Placez vos masques avant de lancer l'entraînement.")

	def case_id(path: Path):
		name = path.name
		stem = name[:-7] if name.endswith(".nii.gz") else path.stem
		if stem.endswith("_0000"):
			return stem[:-5]
		return stem

	image_cases = {case_id(path) for path in image_files}
	label_cases = {case_id(path) for path in label_files}
	missing_labels = sorted(image_cases - label_cases)
	missing_images = sorted(label_cases - image_cases)

	if missing_labels or missing_images:
		parts = []
		if missing_labels:
			parts.append(f"masques manquants pour: {', '.join(missing_labels)}")
		if missing_images:
			parts.append(f"images manquantes pour: {', '.join(missing_images)}")
		raise ValueError(f"Les paires du dataset ne correspondent pas dans {dataset_dir}: {'; '.join(parts)}")

	logging.info("Dataset prêt : %s images et %s masques", len(image_files), len(label_files))


def training_UNet(dataset: int, trainer: str, plan: str, nbfold: int):
	preprocessed = Path(os.environ["nnUNet_preprocessed"])
	dataset_dir_name = dataset_name(dataset)
	reference_dataset_dir_name = dataset_name(DEFAULT_REFERENCE_DATASET_ID)

	try:
		logging.info("Vérification du contenu de nnUNet_raw (%s)", os.environ["nnUNet_raw"])
		run_command(["nnUNetv2_plan_and_preprocess", "-d", str(dataset), "-c", "2d", "--verify_dataset_integrity"])

		src = preprocessed / reference_dataset_dir_name / f"{plan}.json"
		dst_dir = preprocessed / dataset_dir_name
		dst = dst_dir / f"{plan}.json"

		dst_dir.mkdir(parents=True, exist_ok=True)
		if src.exists() and not dst.exists():
			shutil.copy(src, dst)
			logging.info("Plan custom copié : %s -> %s", src, dst)

		run_command(["nnUNetv2_preprocess", "-d", str(dataset), "-c", "2d", "-p", plan, "--clean", "--verify_dataset_integrity"])

		for fold in range(nbfold):
			logging.info("Début Fold %s", fold)
			run_command(["nnUNetv2_train", str(dataset), "2d", str(fold), "-p", plan, "-tr", trainer, "--npz"])
	except Exception as e:
		logging.error("Échec : %s", e)

	try:
		logging.info("Lancement de la crossvalidation")
		run_command(["nnUNetv2_find_best_configuration", str(dataset), "-c", "2d", "-tr", trainer, "-np", "16"])
	except Exception as e:
		logging.error("Echec: %s", e)


def export_nnUnet(dataset, trainer, plan, output):
	try:
		results_path = Path(os.environ["nnUNet_results"])
		dataset_dir_name = dataset_name(dataset)
		model_dir = results_path / dataset_dir_name / f"{trainer}__{plan}__2d"

		logging.info("nnUNet_results = %s", results_path)
		logging.info("Dossier modèle attendu : %s", model_dir)
		logging.info("Existe : %s", model_dir.exists())

		if model_dir.exists():
			logging.info("Contenu : %s", os.listdir(model_dir))

		logging.info("Export zip du model")
		run_command(
			[
				"nnUNetv2_export_model_to_zip",
				"-d",
				str(dataset),
				"-f",
				"0",
				"1",
				"2",
				"3",
				"4",
				"-tr",
				trainer,
				"-p",
				plan,
				"-c",
				"2d",
				"-o",
				str(output),
				"--not_strict",
			]
		)
	except Exception as e:
		logging.error("Échec export : %s", e)


if __name__ == "__main__":
	parser = ArgumentParser(description="Train and export a nnU-Net model.")
	parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT), help="Base directory for nnUNet_raw, nnUNet_preprocessed and nnUNet_results.")
	parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Export destination zip path.")
	args = parser.parse_args()

	setup_environment(args.data_root)
	logging.getLogger().handlers.clear()
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s - %(levelname)s - %(message)s",
		handlers=[
			logging.FileHandler(SCRIPT_DIR / "training_debug.log"),
			logging.StreamHandler(sys.stdout),
		],
	)

	dataset_dir = ensure_dataset_scaffold(DEFAULT_DATASET_ID)
	validate_dataset(dataset_dir)

	Path(args.output).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
	training_UNet(DEFAULT_DATASET_ID, "nnUNetTrainer_250epochs", "nnUNetPlans", 5)
	export_nnUnet(DEFAULT_DATASET_ID, "nnUNetTrainer_250epochs", "nnUNetPlans", Path(args.output).expanduser().resolve())
