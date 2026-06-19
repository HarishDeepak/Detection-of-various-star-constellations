# Rotation-Invariant Star Constellation Recognition

Classical computer vision system for identifying 20 star constellations from images using rotation- and scale-invariant geometric template matching — no deep learning required.

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org)
[![LinkedIn](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/harishdeepak/)

---

## Problem

Given an image of a night sky containing a star constellation at **arbitrary position, scale, and rotation**, identify which of 20 known constellations it belongs to.

**Constellations supported:**
Andromeda, Aquila, Auriga, Canis Major, Capricornus, Cetus, Columba, Gemini, Grus, Leo, Orion, Pavo, Pegasus, Phoenix, Pisces, Piscis Austrinus, Puppis, Ursa Major, Ursa Minor, Vela

---

## Approach

The system avoids deep learning entirely and instead uses **geometric invariant features** — the angular and distance relationships between stars remain constant regardless of how the constellation is rotated or scaled.

### Core Invariance Strategy

For each constellation (template and test image alike):

1. Detect star positions (pixel centroids from contours)
2. **Translate**: shift all coordinates so the brightest star sits at the origin `(0, 0)`
3. **Scale**: divide all coordinates by the distance from the brightest to the second-brightest star (normalises to unit distance)
4. **Rotate**: compute the angle to rotate the second-brightest star to canonical position `(1, 0)` using the law of cosines; apply the same rotation matrix to all stars

After normalisation, two images of the same constellation taken at different orientations and zoom levels produce **nearly identical coordinate sets**.

---

## Processing Pipeline

### Template Building (`makeTemplates()`)

```
Template image (annotated constellation PNG)
  → RGB channel separation
      Red channel   → star blob isolation (dual-threshold subtraction)
      Blue channel  → constellation line isolation
  → Binary thresholding + median filtering (3×3)
  → Canny edge detection
  → Hough Probabilistic Line Transform (detecting constellation lines)
  → cv2.findContours → star centroid extraction
  → Coordinate normalisation (translate → scale → rotate)
  → Serialise (x, y, n_stars, normalised_lines) to pickle file
```

**Channel separation rationale:**
- Red channel captures star blobs (bright round objects)
- Blue channel captures the lines connecting stars in annotated images
- Dual-threshold subtraction (`thresh[0] - thresh[1]`) isolates stars while removing text/labels

### Test Image Recognition (`test_runner()` + `test_normaliser()`)

```
Test image (grayscale night sky)
  → Binary thresholding (190)
  → Median filter (5×5) — noise removal
  → Invert image
  → Canny edge detection
  → cv2.findContours → star centroid extraction
  → For each (brightest, second_brightest) pair permutation:
      Normalise coordinates
  → Load pickle templates
  → Score all constellations via similarity_error()
  → Return constellation with highest score
```

---

## Similarity Scoring

### `similarity_error(train, test)`

For each normalised template, compare against each test permutation:

- For every template star, compute Euclidean distance to all test stars
- Count stars with `min_distance < 0.05` (within tolerance in normalised space)
- Accumulate `error = Σ min_distances` for matched stars

### Scoring Formula

```python
cur_score = count * (count - 2) / (n_stars * error)
```

- **count**: number of matched stars (must be > 2 to avoid noise)
- **n_stars**: total stars in template (penalises partial matches)
- **error**: cumulative matching distance (lower is better fit)
- Capped at 1e+3 to avoid division-by-near-zero artefacts

The permutation loop over all `(brightest, second_brightest)` pairs ensures the correct pair is found even when image quality or star detection order differs.

---

## File Structure

| File / Folder | Contents |
|---|---|
| `Final Project.py` | Full pipeline: template building, test normalisation, scoring, main runner |
| `Templates/` | 20 annotated constellation PNG templates (stars + connecting lines) |
| `test_data/` | 20 test constellation images (one per constellation) |
| `Template Coordinates` | Serialised pickle file of normalised template coordinates |
| `Predicted_images/` | Output images: `true_label predicted_label.png` for each test |

---

## How to Run

**Requirements:** Python 3.x, OpenCV (`cv2`), NumPy, matplotlib, pickle

```bash
pip install opencv-python numpy matplotlib
```

**Step 1 — Build templates** (run once):
```python
# In Final Project.py, uncomment:
makeTemplates()
```
This reads `./Templates/`, processes each constellation, and saves `Template Coordinates`.

**Step 2 — Run test recognition:**
```bash
python "Final Project.py"
```
Tests all 20 constellations in sequence; prints any mismatched predictions and the final accuracy (`count / 20`).

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Classical CV over CNN | Training data for constellation images is sparse; geometric structure is fully deterministic |
| XOR brightness-pair normalisation | Makes matching invariant to unknown orientation in test images |
| ±0.05 tolerance in normalised space | Accommodates contour centroid noise and slight image distortion |
| Permutation over (bright, 2nd-bright) pairs | Handles uncertainty in star detection ordering between template and test |
| Pickle serialisation of templates | Avoids reprocessing template images on every run |

---

## Tech Stack

- **Language**: Python 3
- **Vision**: OpenCV (thresholding, Canny edge detection, Hough transform, contour detection, moments)
- **Geometry**: custom rotation/scale/translation normalisation using `math` (no external geometry library)
- **Utilities**: NumPy, matplotlib, pickle
