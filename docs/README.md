# docs/

| File | What it is |
|---|---|
| `tutorial.html` | The field guide. Open it directly, or serve this directory. Uses a relative path for the montage. |
| `hepatic_montage.png` | Three-panel render used by the tutorial and the top-level README. |
| `hepatic_*.png` | The individual views, written by `examples/render_views.py`. |

The published copy of the tutorial inlines the montage as a data URI so the page
stands alone; that copy is generated, not committed. To rebuild it:

```python
import base64, pathlib
src = pathlib.Path("docs/tutorial.html").read_text()
uri = "data:image/png;base64," + base64.b64encode(
    pathlib.Path("docs/hepatic_montage.png").read_bytes()).decode()
pathlib.Path("tutorial_standalone.html").write_text(
    src.replace('src="hepatic_montage.png"', f'src="{uri}"'))
```

Regenerate the images with:

```bash
python examples/render_views.py SEGMENTATION.nrrd \
    --labels "1=liver,2=hepatic vein,3=portal vein" --primary liver --out docs/hepatic
```
