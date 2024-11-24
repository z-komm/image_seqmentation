

# **SAM-based Image Segmentation**

Dieses Repository bietet ein Python-Skript zur Bildsegmentierung mit dem **Segment Anything Model (SAM)**. Das Skript verwendet SAM, um Masken zu generieren und Bounding-Boxen um erkannte Segmente zu zeichnen. Es ermöglicht die effiziente Analyse und Visualisierung von Segmentierungen in Bildern.

---

## **Inhaltsverzeichnis**
- [Features](#features)
- [Voraussetzungen](#voraussetzungen)
- [Installation](#installation)
- [Verwendung](#verwendung)
- [Beispiele](#beispiele)
- [Hinweise](#hinweise)

---

## **Features**
- **Segmentierung von Bildern**: Identifiziert Segmente und visualisiert Masken.
- **Bounding-Box-Darstellung**: Zeichnet Boxen um erkannte Segmente.
- **Hohe Genauigkeit**: Nutzt SAM-Modelle für präzise Ergebnisse.
- **Flexibilität**: Anpassbare Parameter wie Auflösung, Stabilität und minimale Maskengröße.

---

## **Voraussetzungen**
- **Python-Version**: 3.7 oder höher
- Abhängigkeiten:
  - `opencv-python`
  - `matplotlib`
  - `numpy`
  - `segment-anything`
  - `torch`

---

## **Installation**

1. **Repository klonen**:
   ```bash
   git clone https://github.com/username/sam_segmentation.git
   cd sam_segmentation
   ```

2. **Abhängigkeiten installieren**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Modell herunterladen**:
   Lade das SAM-Modell (z. B. `sam_vit_h_4b8939.pth`) von der [offiziellen SAM-Repository](https://github.com/facebookresearch/segment-anything) herunter und speichere es im Projektverzeichnis.

---

## **Verwendung**

1. **Skript ausführen**:
   ```bash
   python segment_image.py
   ```

2. **Parameter anpassen**:
   Öffne `segment_image.py` und ändere Einstellungen wie:
   - `model_type`: Modellgröße (`vit_b`, `vit_l`, `vit_h`).
   - `points_per_side`, `pred_iou_thresh`, `min_mask_region_area`.

3. **Ergebnisse speichern**:
   - Segmentierte Bilder und Bounding-Boxen werden als PNG-Dateien im Verzeichnis gespeichert.

---

## **Beispiele**

### **Segmentierung**
Originalbild:  
![Original Image](assets/classified_output_third_image.png)

Segmentiertes Bild:  
![Segmented Output](segmented_output.png)

Bounding-Box-Darstellung:  
![Bounding Boxes](segmented_with_boxes.png)

---

## **Hinweise**
- Die Modelldatei muss heruntergeladen und im Projektverzeichnis abgelegt werden.
- Für größere Bilder kann der Speicherverbrauch hoch sein. Stelle sicher, dass genügend GPU-RAM verfügbar ist.

---


