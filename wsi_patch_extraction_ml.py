#Importing libraries
import openslide
import numpy as np
from PIL import Image
import csv
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# Settings
# -----------------------------
slide_path = r"C:\Users\alan\Other\DigitalSlide_B8M_15S_1.mrxs"
output_folder = "tissue_patches"

patch_size = 512
thumbnail_width = 1500
tissue_threshold = 0.20

max_tissue_patches = 50
max_background_patches = 50

os.makedirs(output_folder, exist_ok=True)

# -----------------------------
# Open slide
# -----------------------------
slide = openslide.OpenSlide(slide_path)

print("Dimensions:", slide.dimensions)
print("Levels:", slide.level_count)
print("Level dimensions:", slide.level_dimensions)

width, height = slide.dimensions

# -----------------------------
# Create thumbnail
# -----------------------------
thumbnail_height = int(thumbnail_width * height / width)

thumbnail = slide.get_thumbnail((thumbnail_width, thumbnail_height)).convert("RGB")
thumbnail.save(os.path.join(output_folder, "slide_thumbnail.png"))

thumb_array = np.array(thumbnail)

# -----------------------------
# Detect tissue on thumbnail
# -----------------------------
r = thumb_array[:, :, 0]
g = thumb_array[:, :, 1]
b = thumb_array[:, :, 2]

brightness = thumb_array.mean(axis=2)

colour_difference = (
    np.maximum.reduce([r, g, b]) -
    np.minimum.reduce([r, g, b])
)

# Basic tissue mask:
# Tissue tends to be less white and has more colour variation
tissue_mask = (brightness < 220) & (colour_difference > 10)

mask_image = Image.fromarray((tissue_mask * 255).astype(np.uint8))
mask_image.save(os.path.join(output_folder, "tissue_mask.png"))

# -----------------------------
# Coordinate scaling
# -----------------------------
scale_x = width / thumbnail_width
scale_y = height / thumbnail_height

# How big a full-resolution patch appears on thumbnail
thumb_patch_w = max(1, int(patch_size / scale_x))
thumb_patch_h = max(1, int(patch_size / scale_y))

print("Thumbnail size:", thumbnail.size)
print("Thumbnail patch size:", thumb_patch_w, thumb_patch_h)

# -----------------------------
# Scan slide in patch grid
# -----------------------------
saved_tissue = 0
saved_background = 0

csv_path = os.path.join(output_folder, "patch_log.csv")

with open(csv_path, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    writer.writerow([
        "patch_name",
        "full_x",
        "full_y",
        "tissue_fraction",
        "mean_intensity",
        "colour_difference_mean",
        "label"
    ])

    for full_y in range(0, height - patch_size, patch_size):
        for full_x in range(0, width - patch_size, patch_size):

            # Map full-resolution patch location to thumbnail location
            thumb_x = int(full_x / scale_x)
            thumb_y = int(full_y / scale_y)

            mask_region = tissue_mask[
                thumb_y:thumb_y + thumb_patch_h,
                thumb_x:thumb_x + thumb_patch_w
            ]

            if mask_region.size == 0:
                continue

            tissue_fraction = mask_region.mean()

            # Save tissue-rich examples
            if tissue_fraction >= tissue_threshold and saved_tissue < max_tissue_patches:
                label = 1
                category = "tissue"
                saved_tissue += 1

            # Save low-tissue/background examples
            elif tissue_fraction < 0.05 and saved_background < max_background_patches:
                label = 0
                category = "background"
                saved_background += 1

            else:
                continue

            patch = slide.read_region(
                (full_x, full_y),
                0,
                (patch_size, patch_size)
            ).convert("RGB")

            patch_array = np.array(patch)

            mean_intensity = patch_array.mean()

            r_patch = patch_array[:, :, 0]
            g_patch = patch_array[:, :, 1]
            b_patch = patch_array[:, :, 2]

            colour_difference_patch = (
                np.maximum.reduce([r_patch, g_patch, b_patch]) -
                np.minimum.reduce([r_patch, g_patch, b_patch])
            )

            colour_difference_mean = colour_difference_patch.mean()

            patch_name = f"{category}_patch_x{full_x}_y{full_y}.png"
            patch.save(os.path.join(output_folder, patch_name))

            writer.writerow([
                patch_name,
                full_x,
                full_y,
                round(float(tissue_fraction), 4),
                round(float(mean_intensity), 2),
                round(float(colour_difference_mean), 2),
                label
            ])

            print(
                f"Saved {patch_name} | "
                f"Label: {label} | "
                f"Tissue: {tissue_fraction:.2%} | "
                f"Mean intensity: {mean_intensity:.2f} | "
                f"Colour difference: {colour_difference_mean:.2f}"
            )

            if saved_tissue >= max_tissue_patches and saved_background >= max_background_patches:
                break

        if saved_tissue >= max_tissue_patches and saved_background >= max_background_patches:
            break

print(f"\nFinished. Saved {saved_tissue} tissue patches and {saved_background} background patches.")
print(f"Outputs saved in: {output_folder}")
print(f"Patch log saved as: {csv_path}")

# -----------------------------
# Machine learning section
# -----------------------------
df = pd.read_csv(csv_path)

print("\nDataset summary:")
print(df["label"].value_counts())

features = [
    "tissue_fraction",
    "mean_intensity",
    "colour_difference_mean"
]

X = df[features]
y = df["label"]

# Check that both classes exist before training
if len(y.unique()) < 2:
    print("\nOnly one class found in the dataset. Cannot train classifier.")
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    })

    print("\nFeature importance:")
    print(importance)