# voxelmetry

Physical-unit morphometry and a shareable 3-D viewer for medical segmentations.

Reads medical image volumes, measures the structures in them in millimetres and
litres, and writes an interactive 3-D viewer as a single HTML file. Works on any
integer-labelled segmentation — CT or MR, one structure or thirty.

[`docs/tutorial.html`](docs/tutorial.html) is the field guide: loading,
measuring, viewing, and the traps that show up on real scans. Open it in a
browser or serve `docs/`.

![Liver, hepatic and portal vein trees and five metastases, shown solid, with the liver at 16% opacity, and with the liver hidden](docs/hepatic_montage.png)

The vessel stubs in the middle and right panels are not a rendering fault. At
0.887 mm in-plane against 5 mm between slices, a 3 mm vessel is sampled as
isolated cross-sections; the hepatic vein measures 150 components with the
largest holding 46%. `examples/render_views.py` produces these from the same
meshes the viewer uses.

```bash
voxelmetry view liver_0.nii.gz -o liver.html --labels "1=liver,2=tumour" --split 2 --min-volume 100
```

```
wrote liver.html  (12 structures, 65,532 triangles, 1,569 KB)

structure    volume (mL)  max Ø (mm)  surface (cm²)  sphericity  parts
liver            1359.65       244.2          940.0       0.631      2
tumour_1 *          2.68        20.2            9.8       0.952      1
tumour_2 *          0.89        13.3            4.0       1.000      1
tumour_3 *          0.81        14.5            4.3       0.972      1
tumour_4 *          0.39        12.8            2.3       1.000      1
tumour_5 *          0.36        11.6            2.3       1.000      1
tumour_6 *          0.34        10.3            1.9       1.000      1
tumour_7 *          0.33        12.2            2.1       1.000      1
tumour_8 *          0.24         9.1            2.1       0.882      1
tumour_9 *          0.13         6.9            1.4       0.868      1
tumour_10 *         0.11         6.0            1.2       0.952      1
tumour_11 *         0.11         5.5            1.2       0.923      1

* spans under 5 voxels on its thinnest axis; shape figures are indicative only
```

That output is from MSD Task03 case `liver_0`. Note `parts = 2` on the liver:
the segmentation is not one connected region. Every lesion is flagged because
at 5 mm slice thickness a 6 mm lesion spans barely more than one slice, so its
volume and diameter hold but its shape does not. Neither fact shows up in a
rendering.

---

## Contents

- [Background](#background)
- [Install](#install)
- [Quickstart](#quickstart)
- [The viewer](#the-viewer)
- [Measurements](#measurements)
- [Validation](#validation)
- [Modalities](#modalities)
- [Cohort scan](#cohort-scan)
- [Python API](#python-api)
- [Changes from v1](#changes-from-v1)
- [Layout](#layout)

---

## Background

Started as a 2020 master's project on liver and tumour segmentation from
abdominal CT: preprocess DICOM, segment, stack slices into NRRD, render a
surface, measure it. Those scripts are kept in [`legacy/`](legacy/), with
[notes](legacy/README.md) on what each contributed.

v2 keeps spacing attached to the voxels. A `Volume` carries its millimetres per
voxel through resampling, cropping and meshing, so measurements stay in
physical units instead of drifting whenever the grid changes. Nothing in the
pipeline is organ-specific; it takes whatever integer labels are present, so the
same command handles a liver study, a spleen study or a whole-body segmentation.

## Install

```bash
git clone https://github.com/robert-piecyk/voxelmetry
cd voxelmetry
pip install -e ".[all]"
```

Python 3.10+. Core dependencies are numpy, scipy, scikit-image and SimpleITK.
DICOM reading and quadric mesh decimation are optional extras.

## Quickstart

The phantom generator makes a synthetic torso with an organ, lesions and bone,
so this runs with no data:

```bash
voxelmetry demo -o demo.html
```

With real data:

```bash
voxelmetry info scan.nii.gz
voxelmetry measure segmentation.nii.gz --labels "1=liver,2=tumour" --split 2 --json out.json
voxelmetry view segmentation.nii.gz -o scene.html --labels "1=liver,2=tumour" --split 2
voxelmetry prep dicom_directory/ prepped.nrrd --window abdomen --isotropic 1.0
voxelmetry convert dicom_directory/ volume.nrrd
```

`--split` breaks a label into connected components, so multifocal disease is
measured lesion by lesion rather than as one blob.

### Formats

| Input | Notes |
|---|---|
| NIfTI, NRRD, MetaImage | Single files, read through SimpleITK |
| DICOM series (directory) | Ordered by ImagePositionPatient, never by filename |
| DICOM SEG | `load_dicom_seg()` for a label map, `dicom_seg_masks()` when segments overlap |

CT is assumed to be in Hounsfield units. MR and other uncalibrated data are
detected, and preprocessing steps that need an absolute scale either adapt or
refuse. See [Modalities](#modalities).

### Getting a dataset

Any NIfTI, NRRD, MetaImage or DICOM series works. The
[Medical Segmentation Decathlon](http://medicaldecathlon.com/) tasks have an
adapter that reads label names from `dataset.json`:

```python
from voxelmetry.datasets import MSDDataset, download_command

print(download_command("liver", "./data"))   # prints the curl+tar command
dataset = MSDDataset("./data/Task03_Liver")
image, labels = dataset.cases[0].load()
```

## The viewer

Output is one HTML file with a single external dependency, three.js from a
pinned CDN. Geometry is base64 binary, decoded into typed arrays in the browser.

Drag to orbit, scroll to zoom, shift-drag to pan. The sidebar toggles structures
and shows their volumes; a global opacity slider makes lesions visible through
the organ containing them, and a clip plane cuts along any anatomical axis. A
scale bar tracks the camera. Light and dark themes both work, as does a phone.

A per-structure triangle budget (`--max-faces`) bounds the output size, so it
stays in the hundreds of kilobytes rather than growing with scan resolution.

## Measurements

Per label, and optionally per connected component within a label:

| Quantity | Notes |
|---|---|
| Volume (mm³ and mL) | Voxel count times true voxel volume |
| Max diameter | 3-D Feret diameter, exact over the convex hull |
| Surface area | Marching-cubes triangulation, calibrated against analytic spheres |
| Sphericity | Equivalent-sphere area over measured area; 1.0 is a sphere |
| Centroid, bounding box | Millimetres, in patient coordinates |
| Component count | With the fraction held by the largest, which flags fragmented predictions |

Structures spanning fewer than five voxels on their thinnest axis get
`resolution_limited`. Volume and diameter stay usable; shape descriptors at that
size are dominated by sampling artifacts, so they are reported as indicative
rather than presented as fact.

`lesion_burden()` aggregates components into lesion count, total and largest
volume, largest diameter, and the sum of diameters that RECIST tracks between
timepoints.

## Validation

The phantom generator emits shapes with closed-form geometry, and the tests
assert against that arithmetic rather than against the code's own output:

| Property | Result |
|---|---|
| Sphere volume vs `4/3 πr³` | within 0.07% |
| Sphere surface area vs `4πr²` | within 0.6% |
| Sphere max diameter vs `2r` | exact to the voxel |
| Sphere sphericity | 1.000 ± 0.003 |
| Rod sphericity vs closed form | within 0.08 absolute |
| Volume across three different voxel grids | within 0.5% of each other |
| Mesh-enclosed volume vs voxel count | within 0.5% |
| Volume after 20 Taubin smoothing passes | within 0.5% of unsmoothed |

The last row is why smoothing is Taubin and not Laplacian. Laplacian smoothing
contracts a closed surface slightly on every iteration, so volume measured off a
smoothed mesh drifts downward.

Marching cubes on a hard binary mask traces a staircase and overestimates
surface area by about 9%. A Gaussian pre-smooth of 0.8 voxels holds the error
under 1% across the radii tested. The tests pin the corrected value and the
uncorrected artifact, so a regression in the fix shows up rather than passing
quietly.

```bash
pytest              # 164 tests, including 39 headless checks on the generated viewer
ruff check src tests examples
```

## Modalities

Preprocessing was written for CT, where Hounsfield units put air near −1000.
Nothing else shares that scale. Both failures below were silent until they were
run against a liver MR from TCGA-LIHC with intensities 0 to 831:

| Operation | On non-HU data before | Now |
|---|---|---|
| `body_mask` | Selected 87% of the field of view, since every voxel is above −320 HU | Detects uncalibrated data and falls back to Otsu: 37.9% |
| `apply_window("abdomen")` | Clipped to [−160, 240], collapsing everything above 240 into one value | Raises, and points at `window="percentile"` |

`is_hounsfield()` is the check, `percentile_window()` the alternative for data
with no absolute scale. CT behaviour is unchanged and pinned by a test.

## Cohort scan

`examples/cohort_report.py` measures every case in a dataset in parallel and
reports what disagrees with the rest. On MSD Task03 Liver, 131 cases:

```
slice thickness (mm)  0.70 to 5.00   median 1.00
in-plane spacing (mm) 0.557 to 1.000  median 0.768
anisotropy (z / x)    1.2x median, up to 9.0x

organ volume (mL)     542 to 3195   median 1592   IQR 1380-1850
organ components      1 to 1777   93 case(s) not a single connected region
  stray volume (mm3)  0.0 to 4573.0   median 8.8   worst 0.360% of its organ

cases with lesions    117 of 131
lesions per case      1 to 62   median 3   total 753
lesion burden (%)     0.01 to 45.8   median 0.97
largest lesion Ø (mm) 10 to 241   median 40
```

Slice thickness spans 7x inside the one dataset and anisotropy reaches 9:1. Any
pipeline with kernel sizes in voxels behaves differently across these cases
without saying so, which is why v1's `np.ones((15, 15))` became a millimetre
radius.

71% of liver labels are not a single connected region, but the median stray
volume is 8.8 mm³ and the worst case is 0.36% of its organ. `liver_116` reports
395 components at face-adjacency and 27 at 26-adjacency; 232 of those are single
voxels touching the main body only at a corner. The count reflects the
connectivity choice more than the annotation, so the report leads with stray
volume instead.

Fragmentation correlates with low sphericity at Spearman ρ = −0.58
(p = 2×10⁻¹³), which suggests annotation speckle inflating surface area. That
reading does not hold up: `n_components` and tumour burden correlate at
ρ = +0.85, so heavily diseased livers are both genuinely more irregular and more
fragmented. Controlling for burden drops the partial correlation to −0.32. Some
association remains, but observational data cannot separate speckle from disease
here, and most of the headline effect is confounded.

```bash
python examples/cohort_report.py /path/to/Task03_Liver \
    --organ 1 --lesion 2 --min-lesion-mm3 100 --workers 20 \
    --out outputs/liver_cohort.jsonl
```

Records stream to JSONL and a run resumes from whatever is already on disk, so
an interrupted scan over 131 large volumes keeps its work.

## Python API

```python
import voxelmetry
from voxelmetry.viewer import scene_from_labelmap, write

image  = voxelmetry.load("scan.nii.gz")           # DICOM dir, NIfTI, NRRD, MetaImage
labels = voxelmetry.load("segmentation.nii.gz")

image.spacing          # (5.0, 0.977, 0.977) mm, as (z, y, x)
image.extent_mm        # physical field of view
image.resample(1.0)    # isotropic; extent preserved, spacing updated

liver   = voxelmetry.measure_label(labels, 1, "liver")
tumours = voxelmetry.measure_components(labels, 2, "tumour", min_volume_mm3=50)

print(f"{liver.volume_ml:.0f} mL, {len(tumours)} lesions")
print(voxelmetry.lesion_burden(tumours, reference_volume_ml=liver.volume_ml))

scene = scene_from_labelmap(
    labels,
    names={1: "liver", 2: "tumour"},
    split_labels=(2,),        # one mesh and one row per lesion
    min_component_volume_mm3=50,
)
write(scene, "scene.html")
```

Preprocessing is declarative, so a run can be recorded next to its output:

```python
from voxelmetry.preprocess import PreprocessConfig, run

config = PreprocessConfig(window="abdomen", isotropic_mm=1.0, denoise_mm=0.0)
print(config.describe())
# resample to 1.0 mm isotropic; body mask with 8.0 mm closing; abdomen window
prepped = run(image, config)
```

## Changes from v1

Each of these was a defect with a measurable consequence, and each is now
covered by a test.

| v1 | Consequence | v2 |
|---|---|---|
| Spacing held in a `pydicom` dataset, voxels in a numpy array, then `cv2.resize(img, (256, 256))` | Array reshaped, spacing did not; every later measurement off by the resize factor | `Volume` binds the two and updates spacing on every geometric operation |
| Volume as voxel-fraction of the field of view, then divided by 1e6 and called litres | Wrong by a factor of 1000 | Voxel count times true voxel volume; 1 mL is 1000 mm³ |
| Diameter as the index span of the widest axial slice | Blind to oblique extent and to z entirely | 3-D Feret diameter over the convex hull of the surface voxels |
| `np.ones((15, 15))` for morphological kernels | Closes a different physical gap on every scanner | Structuring elements in millimetres, converted per volume |
| Body mask computed per slice | A slice where the body splits in two loses half the anatomy to the largest-component step | Runs on the whole volume |
| Raw marching cubes into Plotly `create_trisurf` | `temp-plot.html` was 18 MB for one structure | Decimated to a budget, shipped as base64 binary: 902 KB for the five-structure phantom, 1.5 MB for the twelve-structure liver above |

Slice ordering deserves its own note, because it is not a theoretical risk. A
liver CT from TCIA HCC-TACE-Seg is named `00000001.dcm` onward, which sorts
cleanly, yet 46 of its 88 adjacent pairs are out of anatomical order with a mean
index displacement of 31 slices. v1 would have reconstructed noise from it
without raising anything. Ordering now comes from ImagePositionPatient, and a
test writes a deliberately shuffled series to check it.

None of it was runnable elsewhere: every path was `E:/Desktop/`, the patient
list was `range(1, 21)`, and there was no README, no requirements, no tests, and
20 MB of generated artifacts in git. Paths are arguments now, dependencies are
declared, and CI runs the suite on three Python versions.

Two things v1 imported but never contained: a U-Net (Keras layers are imported
in every file, but no model is defined) and any evaluation (`evaluate` is
commented out). Segmentation is out of scope here — this package consumes
segmentations. Producing them is the next piece of work.

## Layout

```
src/voxelmetry/
├── volume.py        Volume: array + spacing + origin, geometry-preserving ops
├── io.py            DICOM series, DICOM SEG, NIfTI, NRRD; the (x,y,z)/(z,y,x) flip
├── preprocess.py    HU windowing, body extraction, denoising, declarative config
├── measure.py       Volumetry, Feret diameter, surface area, sphericity, burden
├── mesh.py          Marching cubes, quadric decimation, Taubin smoothing
├── phantom.py       Synthetic volumes with closed-form geometry
├── cli.py           measure · view · prep · convert · info · demo
├── viewer/
│   ├── scene.py     Label-agnostic scene assembly and palette
│   └── html.py      Self-contained page: base64 geometry, inline three.js viewer
└── datasets/
    └── msd.py       Medical Segmentation Decathlon adapter
```

## Licence

MIT. Datasets carry their own terms; MSD tasks are CC-BY-SA 4.0.
