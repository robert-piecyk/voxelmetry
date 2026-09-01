# docs/

Published to GitHub Pages by `.github/workflows/pages.yml` on any push to
`main` that touches this directory. The directory is served as-is; there is no
build step, so `index.html` must keep referencing its images relatively.

| File | What it is |
|---|---|
| `index.html` | The field guide, and the Pages landing page |
| `hepatic_montage.png` | Three-panel render used by the guide and the top-level README |
| `hepatic_{solid,ghost,inner,clip}.png` | The individual views |

Regenerate the images from any segmentation:

```bash
python examples/render_views.py SEGMENTATION.nrrd \
    --labels "1=liver,2=hepatic vein,3=portal vein" --primary liver --out docs/hepatic
```

To open the guide without a server, `index.html` works from the filesystem as
long as the PNGs sit beside it. For a single self-contained file, inline the
montage:

```python
import base64, pathlib
src = pathlib.Path("docs/index.html").read_text()
uri = "data:image/png;base64," + base64.b64encode(
    pathlib.Path("docs/hepatic_montage.png").read_bytes()).decode()
pathlib.Path("field-guide.html").write_text(
    src.replace('src="hepatic_montage.png"', f'src="{uri}"'))
```
