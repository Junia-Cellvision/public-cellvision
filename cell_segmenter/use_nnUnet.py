import os
import torch
import shutil
import zipfile
import tempfile

def nnUnet_predict(zip_path,img_folder, pred_folder): 
    tmp_dir = tempfile.mkdtemp()
    
    try:
        print(f"--- Extraction du ZIP : {zip_path} ---")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(tmp_dir)
        
        model_folder = None
        for root, dirs, files in os.walk(tmp_dir):
            if any(d.startswith('fold_') for d in dirs) and 'plans.json' in files:
                model_folder = root
                break
        
        if not model_folder:
            print("ERREUR : Impossible de trouver un dossier avec des folds et plans.json.")
            return

        target_dataset_json = os.path.join(model_folder, 'dataset.json')
        if not os.path.exists(target_dataset_json):
            print("dataset.json manquant dans le dossier cible, recherche globale...")
            found = False
            for root, dirs, files in os.walk(tmp_dir):
                if 'dataset.json' in files:
                    source_path = os.path.join(root, 'dataset.json')
                    shutil.copy(source_path, target_dataset_json)
                    print(f"SUCCÈS : dataset.json copié de {root} vers {model_folder}")
                    found = True
                    break
            if not found:
                print("ERREUR CRITIQUE : dataset.json introuvable dans toute l'archive ZIP.")
                return
        
        print(f"Contenu final du dossier modèle : {os.listdir(model_folder)}")
        
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        predictor = nnUNetPredictor(
            tile_step_size=0.3, 
            use_gaussian=True,
            use_mirroring=True, 
            perform_everything_on_device=True,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            verbose=False,
            allow_tqdm=True
        )

        print("Initialisation des poids (ceci peut prendre un moment)...")
        predictor.initialize_from_trained_model_folder(
            model_folder,
            use_folds=None, # Will be auto-detected
        )
        
        print(f"Lancement de la segmentation sur {img_folder}")
        os.makedirs(pred_folder, exist_ok=True)
        predictor.predict_from_files(
            img_folder,
            pred_folder,
            save_probabilities=False,
            overwrite=True,
            num_processes_preprocessing=12, 
            num_processes_segmentation_export=8
        )
        print(f"PRÉDICTION TERMINÉE. Résultats dans : {pred_folder}")

        # TODO Else, we use the cli -> simpler maintenance ?
        # subprocess.run([
        #     "nnUNetv2_predict_from_modelfolder",
        #     "-i", img_folder,
        #     "-o", pred_folder,
        #     "-m", model_folder,
        #     "-f", "0", "1", "2", "3", "4",
        #     "-device", "cuda"
        # ], check=True)

    except Exception as e:
        print(f"ÉCHEC DU SCRIPT : {e}")
    finally:
        print("Nettoyage des fichiers temporaires...")
        shutil.rmtree(tmp_dir)

