from PIL import Image
import torch
from ultralytics import YOLO


class ElectrodeSegmenter:
    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.model = YOLO(model_path)

    def segment(
        self, img_data: Image.Image, conf=0.25, iou=0.45, device: torch.device|None = None
    ):  
        electrode_ends_box: list[tuple[float, float, float, float]] = []
        electrode_bars_poly: list[list[tuple[float, float]]] = []

        if self.model is None:
            raise ValueError("No model loaded. Pass model_path to __init__.")

        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        results = self.model.predict(
            source=img_data,
            conf=conf,
            iou=iou,
            save=False,
            verbose=False,
            device=device,
            imgsz=640,
            retina_masks=False,
        )

        for r in results:
            r = r.cpu()
            if r.boxes is None:
                continue
            for idx, cls_tensor in enumerate(r.boxes.cls):
                cls_id = int(cls_tensor.item())
                class_name = r.names[cls_id]
                if class_name == "electrode_bar" or cls_id == 0:
                    if r.masks is not None and idx < len(r.masks.xy):
                        poly = r.masks.xy[idx].tolist()
                        electrode_bars_poly.append(poly)
                elif class_name == "electrode_end" or cls_id == 1:
                    box = r.boxes.xyxy[idx].tolist()
                    electrode_ends_box.append(tuple(box))

        return electrode_ends_box, electrode_bars_poly
