import cv2
import numpy as np

# Load the image
image_path = "Figure_1.png"
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to HSV to detect red boxes
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define red color range in HSV
lower_red1 = np.array([0, 50, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 50, 50])
upper_red2 = np.array([180, 255, 255])

# Mask for red color
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
red_mask = cv2.bitwise_or(mask1, mask2)

# Find contours of the red boxes
contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Classify based on position and size
height, width, _ = image.shape
classified_areas = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    area_center_x = x + w / 2
    area_center_y = y + h / 2

    # Define classification logic
    if w > width * 0.6 and h > height * 0.1:  # Large central area
        category = "Main Text Block"
    elif x < width * 0.2:  # Left-aligned for page numbers
        category = "Page Number"
    elif y < height * 0.2 or y > height * 0.8:  # Margins for annotations
        category = "Marginal Notes"
    else:
        category = "Unknown"

    # Store the classification
    classified_areas.append((x, y, w, h, category))

    # Draw bounding boxes with labels
    color = (0, 255, 0) if category == "Main Text Block" else (255, 0, 0)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        image,
        category,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )

# Save and display the classified image
output_path = "classified_output.png"
cv2.imwrite(output_path, image)
print(f"Classified image saved as {output_path}")

# Display classifications
for area in classified_areas:
    print(f"Area at (x: {area[0]}, y: {area[1]}, w: {area[2]}, h: {area[3]}): {area[4]}")
