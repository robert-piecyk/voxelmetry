"""Render static views of a scene, for READMEs and slides.

Draws the same meshes as the interactive viewer, through the same scene
assembly, for contexts that cannot embed one.

    python examples/render_views.py segmentation.nii.gz \\
        --labels "1=liver,2=tumour" --split 2 --out docs/liver

Rendering is offscreen via VTK, so it needs no display. Requires pyvista.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from voxelmetry.io import load
from voxelmetry.mesh import Mesh
from voxelmetry.viewer import Scene, scene_from_labelmap

BACKGROUND = "#14171a"
CAPTION = "#c8ccd0"


def to_polydata(mesh: Mesh):
    """Convert a :class:`~voxelmetry.mesh.Mesh` to PyVista's face-array layout."""
    import pyvista as pv

    faces = np.hstack(
        [np.full((mesh.n_faces, 1), 3, np.int64), mesh.faces.astype(np.int64)]
    ).ravel()
    return pv.PolyData(mesh.vertices.astype(np.float64), faces)


def render(
    scene: Scene,
    out_prefix: Path,
    primary: str | None = None,
    size: tuple[int, int] = (1100, 900),
    azimuth: float = 25.0,
    elevation: float = 18.0,
    captions: bool = False,
) -> list[Path]:
    """Write four views: solid, translucent, interior only, and clipped.

    Args:
        scene: The scene to draw.
        out_prefix: Path prefix; ``_solid.png`` and friends are appended.
        primary: Name of the enclosing structure, made translucent or hidden so
            the others show. Defaults to the largest by volume.
        size: Pixel dimensions per view.
        azimuth: Camera rotation about the vertical, in degrees.
        elevation: Camera elevation, in degrees.
        captions: Burn the caption into the image. Off by default, because a
            caption anchored to the canvas defeats the content-trimming the
            montage does -- it labels its panels itself instead.

    Returns:
        The paths written.
    """
    import pyvista as pv

    pv.OFF_SCREEN = True
    if not scene.meshes:
        raise ValueError("scene has no meshes to render")

    if primary is None:
        primary = max(scene.meshes, key=lambda m: m.metadata.get("volume_ml", 0.0)).name

    prepared = [(m.name, m.color, to_polydata(m)) for m in scene.meshes]
    enclosing = next((p for p in prepared if p[0] == primary), prepared[0])

    views = [
        ("solid", "all structures, full opacity"),
        ("ghost", f"{primary} at 16% opacity"),
        ("inner", f"{primary} hidden"),
        ("clip", "clip plane through the midpoint"),
    ]
    written = []
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    for mode, caption in views:
        plotter = pv.Plotter(off_screen=True, window_size=size)
        plotter.set_background(BACKGROUND)

        for name, color, poly in prepared:
            is_primary = name == primary
            if mode == "inner" and is_primary:
                continue
            if mode == "clip":
                poly = poly.clip("z", origin=enclosing[2].center, invert=True)
                if poly.n_points == 0:
                    continue
            opacity = 0.16 if (is_primary and mode == "ghost") else (
                0.30 if (is_primary and mode == "clip") else 1.0
            )
            plotter.add_mesh(
                poly, color=color, opacity=opacity, smooth_shading=True, specular=0.28
            )

        plotter.camera_position = "xz"
        plotter.camera.azimuth = azimuth
        plotter.camera.elevation = elevation
        plotter.enable_lightkit()
        if captions:
            plotter.add_text(caption, position="lower_left", font_size=11, color=CAPTION)

        path = out_prefix.with_name(f"{out_prefix.name}_{mode}.png")
        plotter.screenshot(str(path))
        plotter.close()
        written.append(path)

    return written


def montage(
    paths: list[Path],
    out: Path,
    captions: list[str] | None = None,
    columns: int = 3,
    gutter: int = 14,
) -> Path:
    """Tile rendered views into one image, each trimmed to its own content.

    Panels are cropped to their subject before tiling, so a small structure does
    not end up surrounded by background.

    Args:
        paths: Rendered PNGs, in order.
        out: Destination image.
        captions: One label per panel, drawn beneath it.
        columns: How many panels to include.
        gutter: Pixels between panels.

    Returns:
        The path written.
    """
    from PIL import Image, ImageDraw

    background = (20, 23, 26)
    ink = (200, 204, 208)

    trimmed = []
    for path in paths[:columns]:
        image = Image.open(path).convert("RGB")
        array = np.asarray(image).astype(int)
        mask = np.abs(array - np.array(background)).sum(axis=2) > 24
        rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
        if len(rows) and len(cols):
            pad = 18
            image = image.crop((
                max(int(cols[0]) - pad, 0), max(int(rows[0]) - pad, 0),
                min(int(cols[-1]) + pad, image.width), min(int(rows[-1]) + pad, image.height),
            ))
        trimmed.append(image)

    height = min(im.height for im in trimmed)
    scaled = [im.resize((round(im.width * height / im.height), height)) for im in trimmed]

    strip = 34 if captions else 0
    width = sum(im.width for im in scaled) + gutter * (len(scaled) - 1)
    sheet = Image.new("RGB", (width, height + strip), background)
    draw = ImageDraw.Draw(sheet)

    x = 0
    for index, image in enumerate(scaled):
        sheet.paste(image, (x, 0))
        if captions and index < len(captions):
            draw.text((x + 4, height + 9), captions[index], fill=ink)
        x += image.width + gutter

    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, optimize=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labelmap", type=Path, help="Segmentation volume.")
    parser.add_argument("--labels", default=None, help="Names, e.g. '1=liver,2=tumour'.")
    parser.add_argument("--split", default=None, help="Labels to split into components.")
    parser.add_argument("--primary", default=None, help="Structure to make translucent.")
    parser.add_argument("--max-faces", type=int, default=40_000)
    parser.add_argument("--out", type=Path, default=Path("docs/scene"))
    args = parser.parse_args(argv)

    from voxelmetry.cli import _parse_labels, _parse_split

    labelmap = load(args.labelmap)
    scene = scene_from_labelmap(
        labelmap,
        names=_parse_labels(args.labels),
        split_labels=_parse_split(args.split),
        max_faces_per_structure=args.max_faces,
    )
    paths = render(scene, args.out, primary=args.primary)
    for path in paths:
        print("wrote", path)
    labels = ["all structures", f"{args.primary or 'organ'} at 16% opacity",
              f"{args.primary or 'organ'} hidden", "clipped at the midpoint"]
    print("wrote", montage(
        paths, args.out.with_name(f"{args.out.name}_montage.png"), captions=labels
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
