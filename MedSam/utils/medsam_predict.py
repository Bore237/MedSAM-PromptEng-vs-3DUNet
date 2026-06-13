import os
import numpy as np
import torch
from segment_anything import SamPredictor, sam_model_registry
from utils import MedSamMetrics
import time

class MedSamWrapper:
    def __init__(self, model_type="vit_b", checkpoint_path=None, device="cpu"):
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.predictor = None

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Chargement de MedSAM sur : {device}")
            sam = sam_model_registry[model_type]()
            state_dict = torch.load(checkpoint_path, map_location=torch.device(device))
            sam.load_state_dict(state_dict)
            sam.to(device)
            sam.eval()
            self.predictor = SamPredictor(sam)
        else:
            raise FileNotFoundError(
                f"Checkpoint introuvable : {checkpoint_path}\n"
                "Téléchargez le modèle depuis https://github.com/bowang-lab/MedSAM"
            )

    def clear_image(self, verbose=0):
        """Réinitialise l'image dans le prédicteur si déjà définie."""
        if self.predictor and self.predictor.is_image_set:
            self.predictor.reset_image()
            if verbose > 0:
                print("Image reset:", self.predictor.is_image_set)

    def preprocess_slice(self, img_slice):
        """Normalise une coupe en [0,255] et convertit en RGB."""
        slide_norm = ((img_slice - img_slice.min()) / (img_slice.max() - img_slice.min()) * 255).astype(np.uint8)
        image_rgb = np.stack([slide_norm, slide_norm, slide_norm], axis=-1)
        return image_rgb

    def predict(self, img_slice, mask_ref, input_box=None, points=None, labels=None, multimask=True):
        """Applique SAM sur une coupe et calcule les métriques."""
        image_rgb = self.preprocess_slice(img_slice)
        self.clear_image()
        self.predictor.set_image(image_rgb)
        start_time = time.time()

        masks_sam, scores, logits = self.predictor.predict(
            box=input_box,
            point_coords=points,
            point_labels=labels,
            multimask_output=multimask,
        )

        inference_time = time.time() - start_time
        metrics = MedSamMetrics(img_slice, mask_ref, masks_sam, scores)
        dice = metrics.dice()
        iou = metrics.iou()

        return {
            "masks": masks_sam,
            "scores": scores,
            "logits": logits,
            "dice": dice,
            "iou": iou,
            "time": inference_time,  
        }
