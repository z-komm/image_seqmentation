import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Bildpfad laden
image_path = "assets/classified_output_third_image.png"
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# SAM-Modell laden (Modell und Gewichtspfad anpassen)
model_type = "vit_h"  # Optionen: "vit_b", "vit_l", "vit_h"
checkpoint_path = "sam_vit_h_4b8939.pth"  # Lade dies von der offiziellen SAM-Repository
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)

# Automatic Mask Generator mit erweiterten Parametern initialisieren
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=32,  # Erhöht die Maskenauflösung
    pred_iou_thresh=0.9,  # Höhere Genauigkeit der Masken
    stability_score_thresh=0.9,  # Nur stabile Masken verwenden
    min_mask_region_area=1000,  # Minimale Maskengröße (in Pixeln)
)

# Segmentierung durchführen
masks = mask_generator.generate(image)

# Ergebnisse visualisieren und Masken speichern
def show_anns(image, masks):
    if len(masks) == 0:
        print("Keine Segmente gefunden!")
        return
    
    annotated_image = image.copy()
    for mask in masks:
        # Zeichne jede Maske grün und speichere sie separat
        annotated_image[mask["segmentation"]] = [0, 255, 0]
        
    # Zeige die annotierte Version
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_image)
    plt.axis("off")
    plt.show()

    # Optional: Speichere das annotierte Bild
    cv2.imwrite("segmented_output.png", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

# Anzeigen der Segmente
show_anns(image, masks)

# Bounding-Boxen zeichnen
def draw_bounding_boxes(image, masks):
    boxed_image = image.copy()
    for mask in masks:
        # Berechne die Bounding-Box der Maske
        coords = np.where(mask["segmentation"])
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        # Zeichne die Bounding-Box
        cv2.rectangle(boxed_image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)  # Blau für Boxen
    
    plt.figure(figsize=(10, 10))
    plt.imshow(boxed_image)
    plt.axis("off")
    plt.show()
    
    # Optional: Speichere das Bild mit Bounding-Boxen
    cv2.imwrite("segmented_with_boxes.png", cv2.cvtColor(boxed_image, cv2.COLOR_RGB2BGR))

# Bounding-Boxen zeichnen
draw_bounding_boxes(image, masks)
