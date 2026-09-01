# legacy/

The original 2020 scripts, kept unmodified for provenance. They are not
importable, not tested and not maintained; `src/voxelmetry/` supersedes them.

They are preserved because they record the problem being solved, and because
several of the ideas in v2 started here:

| Script | What it contributed |
|---|---|
| `main.py`, `2607.py` | The DICOM preprocessing recipe: HU clipping, denoising, histogram equalisation, Otsu thresholding, largest-connected-component body extraction. Ported to 3D in `voxelmetry/preprocess.py`. |
| `3007 visual.py`, `example visualisation 3d using nrrd images.py` | Stacking per-patient slices into a single NRRD. Superseded by `voxelmetry/io.py`, which reads the series directly and keeps its spacing. |
| `example visualisation 30.07.py` | Marching-cubes surface extraction, resampling to isotropic voxels, and the volume/extent measurements. Rebuilt in `voxelmetry/mesh.py` and `voxelmetry/measure.py`. |
| `example thickness and so on.py` | Exploratory work on thresholding and body-mask refinement. |
| `app test.py`, `example TKinter*.py` | Unmodified Tkinter tutorial snippets; the GUI direction was abandoned in favour of the browser viewer. |

Running any of them today would fail: they hard-code `E:/Desktop/` paths and
call APIs that have since been removed (`marching_cubes_lewiner`,
`plotly.tools.FigureFactory`, `fig.gca(projection='3d')`,
`scipy.ndimage.interpolation.zoom`, standalone `keras`).
