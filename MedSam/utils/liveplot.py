import matplotlib.pyplot as plt
from IPython.display import clear_output

class MetricsLivePlot:
    """
    Live plot for segmentation metrics (Dice & IoU) over slices,
    with one column per modality.
    Works in VSCode/Jupyter even without %matplotlib widget.
    """

    def __init__(self, modalities, figsize=(18, 5), max_y=1.05):
        self.modalities = modalities
        self.n_mod = len(modalities)
        self.figsize = figsize
        self.max_y = max_y

        # Stockage des données
        self.plots = {}
        for mod in modalities:
            self.plots[mod] = {
                "dice_x": [], "dice_y": [],
                "iou_x": [], "iou_y": []
            }

    def update(self, modality, index, dice, iou):
        # Ajouter les nouvelles valeurs
        p = self.plots[modality]
        p["dice_x"].append(index)
        p["dice_y"].append(dice)
        p["iou_x"].append(index)
        p["iou_y"].append(iou)

        # Effacer l'ancienne figure
        clear_output(wait=True)
        fig, axes = plt.subplots(1, self.n_mod, figsize=self.figsize)
        if self.n_mod == 1:
            axes = [axes]

        # Replot pour chaque modalité
        for idx, mod in enumerate(self.modalities):
            ax = axes[idx]
            ax.plot(self.plots[mod]["dice_x"], self.plots[mod]["dice_y"], label="Dice", color="blue")
            ax.plot(self.plots[mod]["iou_x"], self.plots[mod]["iou_y"], label="IoU", color="orange")
            ax.set_title(f"Live Metrics – {mod}")
            ax.set_xlabel("Slice index")
            ax.set_ylabel("Metric value")
            ax.set_ylim(0, self.max_y)
            ax.set_xlim(0, max(5, index))
            ax.legend()

        plt.tight_layout()
        plt.show()
