# Digital-pathology-openslide-workflow
Python and OpenSlide workflow for whole-slide image tissue detection, patch extraction and basic ML classification

This project is a small exploratory Python workflow for working with whole-slide pathology images using OpenSlide.

The script demonstrates how to:

- Load a whole-slide image file (`.mrxs`)
- Inspect slide dimensions and resolution levels
- Generate a low-resolution thumbnail
- Create a simple tissue mask using brightness and colour variation
- Map thumbnail coordinates back to full-resolution slide coordinates
- Extract tissue-rich and background image patches
- Save extracted image tiles
- Record patch-level metadata in a CSV file
- Train a basic Random Forest classifier using simple image features

## Purpose

This project was created as a learning and demonstration workflow for digital pathology and computational pathology. It is intended to show how Python can be used to process large pathology image files and prepare image patches for downstream analysis.

## Tools Used

- Python
- OpenSlide
- NumPy
- Pillow
- pandas
- scikit-learn

## Workflow Summary

1. Open a whole-slide pathology image using OpenSlide.
2. Generate a thumbnail of the full slide.
3. Identify likely tissue-containing regions using simple colour and brightness thresholds.
4. Extract 512 x 512 pixel patches from tissue-rich and low-tissue/background regions.
5. Save patch metadata including:
   - patch name
   - full-resolution x/y coordinates
   - tissue fraction
   - mean intensity
   - colour difference
   - tissue/background label
6. Train a simple Random Forest classifier to distinguish tissue-rich patches from low-tissue/background patches.

## Example Output

The script creates an output folder called:

```text
tissue_patches
