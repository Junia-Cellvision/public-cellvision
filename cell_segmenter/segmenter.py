from PIL import Image
import shutil
import tempfile
from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from cell_segmenter.use_nnUnet import nnUnet_predict
import pandas as pd
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from scipy import ndimage


from cell_segmenter.use_nnUnet import nnUnet_predict


DATA_FOLDER = Path(__file__).parent.parent / "data/cell_segmenter/"
MODEL_PATH = DATA_FOLDER / "NNunet_full_model.zip"
for key in ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']:
    os.environ[key] = os.path.join(DATA_FOLDER, key)

def _setup_nnunet_dirs():
    for key in ('nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results'):
        path = DATA_FOLDER / key
        os.environ[key] = str(path)
        if path.exists():
            shutil.rmtree(path,ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

_setup_nnunet_dirs()

class CellSegmenter:
    def __init__(self, image_path, output_dir="../output"):
        self.output_dir = Path(output_dir)
        self.tmpdir = Path(tempfile.mkdtemp())
        
        self.source = Path(self.tmpdir) / (Path(image_path).stem + "_0000.png")
        img = Image.open(image_path).convert("L")
        img.save(self.source, format="PNG")

        print(f"tmpdir: {self.tmpdir}")
        print(f"output_dir: {self.output_dir}")
        folder = Path(output_dir)
        for item in folder.iterdir():
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item)
            
    def segment(self):
        zip_path = str(MODEL_PATH)
        input_dir = str(self.tmpdir)
        output_dir = str(self.tmpdir)
        nnUnet_predict(zip_path, input_dir, output_dir)
        for f in os.listdir(self.tmpdir):
            print(f)

    def Mix(self, show=True, alpha=0.45):
        for f in sorted(os.listdir(self.tmpdir)):
            if not f.endswith("_0000.png"):
                continue
            img_name = f
            msk_name = f.replace("_0000.png", ".png")
            path_img = os.path.join(self.tmpdir, img_name)
            path_msk = os.path.join(self.tmpdir, msk_name)
            if not os.path.exists(path_msk):
                print(f"⚠️  Masque manquant pour {img_name}")
                continue

            img_in  = np.array(Image.open(path_img))
            img_out = np.array(Image.open(path_msk))

            img = img_in.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

            n_labels = int(img_out.max()) + 1
            rng = np.random.default_rng(42)
            colors = rng.random((n_labels, 3))
            colors[0] = 0

            mask_rgb = colors[img_out]
            fg = img_out > 0
            overlay = img.copy()
            overlay[fg] = (1 - alpha) * img[fg] + alpha * mask_rgb[fg]

            # --- Affichage ---
            fig, ax = plt.subplots(1, 3, figsize=(20, 6))
            ax[0].imshow(img_in, cmap='gray')
            ax[0].set_title(f"IMAGE: {img_name}")
            ax[0].axis('off')
            ax[1].imshow(img_out, cmap='nipy_spectral')
            ax[1].set_title(f"MASQUE: {msk_name}")
            ax[1].axis('off')
            ax[2].imshow(overlay)
            ax[2].set_title("OVERLAY")
            ax[2].axis('off')
            plt.tight_layout()

            # --- Sauvegarde dans output_dir ---
            fig_path = self.output_dir / "overlay.png"
            plt.savefig(fig_path, dpi=120, bbox_inches='tight')

            if show:
                plt.show()
            else:
                plt.close(fig)

    def clearAndCopy(self):
        for f in os.listdir(self.tmpdir):
            src = self.tmpdir / f
            if f.endswith("_0000.png"):
                shutil.copy2(src, self.output_dir / "input.png")
            elif f.endswith(".png"):
                shutil.copy2(src, self.output_dir / "mask.png")
        shutil.rmtree(self.tmpdir)
        print(f"Fichiers copiés vers {self.output_dir}, tmpdir supprimé.")

    def Metrics(
        self,
        target_class=1,
        connectivity=8,
        min_area=0,
        alpha=0.55,
        show=True,
    ):
        """
        Pour chaque masque produit par nnUNet, calcule les composantes
        connexes de `target_class` et exporte :
          - <mask>_metrics.json : un JSON (id, aire, centroïde, bbox) par
                                  composante + métadonnées globales.
          - <mask>_metrics.png  : overlay coloré sur l'image originale
                                  avec bbox, centroïde et étiquette par
                                  composante.
 
        Retourne un dict {nom_du_masque: pd.DataFrame}.
        """
        if connectivity == 4:
            structure = ndimage.generate_binary_structure(2, 1)
        elif connectivity == 8:
            structure = ndimage.generate_binary_structure(2, 2)
        else:
            raise ValueError("connectivity doit être 4 ou 8")
 
        results = {}
 
        for f in sorted(os.listdir(self.tmpdir)):
            if not f.endswith("_0000.png"):
                continue
            img_name = f
            msk_name = f.replace("_0000.png", ".png")
            path_img = self.tmpdir / img_name
            path_msk = self.tmpdir / msk_name
            if not path_msk.exists():
                print(f"⚠️  Masque manquant pour {img_name}")
                continue
 
            # --- chargement -------------------------------------------------
            img_in = np.array(Image.open(path_img))
            mask = np.array(Image.open(path_msk))
            if mask.ndim == 3:
                mask = mask[:, :, 0]
 
            binary = (mask == target_class).astype(np.uint8)
            labels, n_comp = ndimage.label(binary, structure=structure)
 
            # --- métriques --------------------------------------------------
            masks_dir = self.output_dir / "masks"
            masks_dir.mkdir(exist_ok=True)

            rows = []
            for i in range(1, n_comp + 1):
                ys, xs = np.where(labels == i)
                area = ys.size
                if area < min_area:
                    labels[labels == i] = 0  # on l'enlève aussi de la viz
                    continue
                rows.append({
                    'component_id': i,
                    'area_px': int(area),
                    'centroid_y': round(float(ys.mean()), 1),
                    'centroid_x': round(float(xs.mean()), 1),
                    'bbox_y_min': int(ys.min()),
                    'bbox_x_min': int(xs.min()),
                    'bbox_y_max': int(ys.max()),
                    'bbox_x_max': int(xs.max()),
                    'bbox_h': int(ys.max() - ys.min() + 1),
                    'bbox_w': int(xs.max() - xs.min() + 1),
                })
                cell_mask = (labels == i).astype(np.uint8) * 255
                Image.fromarray(cell_mask).save(masks_dir / f"cell_{i:04d}.png")
 
            df = (
                pd.DataFrame(rows)
                .sort_values('area_px', ascending=False)
                .reset_index(drop=True)
            )

            # Renumber cells 1..N in sorted order and rename mask files accordingly.
            old_ids = df['component_id'].tolist()
            for old_id in old_ids:
                p = masks_dir / f"cell_{int(old_id):04d}.png"
                if p.exists():
                    p.rename(masks_dir / f"_tmp_{int(old_id):04d}.png")
            for new_id, old_id in enumerate(old_ids, start=1):
                p = masks_dir / f"_tmp_{int(old_id):04d}.png"
                if p.exists():
                    p.rename(masks_dir / f"cell_{new_id:04d}.png")
            df['component_id'] = range(1, len(df) + 1)

            # --- export JSON ------------------------------------------------
            json_path = self.output_dir / "metrics.json"
            with open(json_path, 'w', encoding='utf-8') as fp:
                json.dump({
                    'mask': msk_name,
                    'target_class': target_class,
                    'connectivity': connectivity,
                    'min_area': min_area,
                    'n_components': len(df),
                    'class_pixels': int(binary.sum()),
                    'components': df.to_dict(orient='records'),
                }, fp, indent=2)
 
            # --- visualisation colorée -------------------------------------
            H, W = mask.shape
            n_show = max(len(df), 1)
            rng = np.random.default_rng(42)
            colors = plt.cm.tab20(np.linspace(0, 1, n_show))
            rng.shuffle(colors)
            cmap = ListedColormap(np.vstack([[0, 0, 0, 0], colors]))
 
            fig, ax = plt.subplots(figsize=(12, 12 * H / W))
            ax.imshow(img_in, cmap='gray')
            ax.imshow(
                np.ma.masked_where(labels == 0, labels),
                cmap=cmap, alpha=alpha, vmin=0, vmax=n_comp,
            )
 
            for _, row in df.iterrows():
                x0, y0 = row['bbox_x_min'], row['bbox_y_min']
                w, h = row['bbox_w'], row['bbox_h']
                ax.add_patch(mpatches.Rectangle(
                    (x0, y0), w, h, fill=False,
                    edgecolor='white', linewidth=1.2,
                ))
                ax.plot(row['centroid_x'], row['centroid_y'],
                        'wx', markersize=8, mew=1.5)
                ax.text(
                    x0, y0 - 6,
                    f"#{row['component_id']}  A={row['area_px']}",
                    color='white', fontsize=9,
                    bbox=dict(facecolor='black', alpha=0.6,
                              pad=2, edgecolor='none'),
                )
 
            ax.set_title(
                f"{msk_name} — classe {target_class} — {len(df)} composante(s) — "
                f"{int(binary.sum()):,} px",
                fontsize=12,
            )
            ax.set_xlabel('x (colonne)')
            ax.set_ylabel('y (ligne)')
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
            plt.tight_layout()
 
            fig_path = self.output_dir / "metrics.png"
            plt.savefig(fig_path, dpi=120, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)
 
            print(f"[{msk_name}] {len(df)} composantes → {json_path.name}, {fig_path.name}")
            #results[msk_name] = df

        print(results)
        return results