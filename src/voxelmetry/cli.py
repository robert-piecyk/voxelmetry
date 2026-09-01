"""Command-line interface.

Paths and options are arguments; nothing is written outside the directory named
by the caller.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import io as nio
from . import measure as nmeasure
from .preprocess import HU_WINDOWS, PreprocessConfig
from .preprocess import run as preprocess_run
from .viewer import html as nhtml
from .viewer import scene_from_labelmap

app = typer.Typer(
    add_completion=False,
    help="Morphometry for CT and MR segmentations, with a 3-D viewer that fits in one HTML file.",
)
console = Console()


def _parse_labels(spec: str | None) -> dict[int, str]:
    """Parse ``1=liver,2=tumour`` into a label-name mapping.

    Raises:
        typer.BadParameter: If the spec is malformed.
    """
    if not spec:
        return {}
    mapping: dict[int, str] = {}
    for chunk in spec.split(","):
        if "=" not in chunk:
            raise typer.BadParameter(f"expected value=name, got {chunk!r}")
        value, name = chunk.split("=", 1)
        try:
            mapping[int(value.strip())] = name.strip()
        except ValueError as exc:
            raise typer.BadParameter(f"label value must be an integer, got {value!r}") from exc
    return mapping


def _parse_split(spec: str | None) -> tuple[int, ...]:
    if not spec:
        return ()
    try:
        return tuple(int(v) for v in spec.split(",") if v.strip())
    except ValueError as exc:
        raise typer.BadParameter(f"--split expects comma-separated integers, got {spec!r}") from exc


def _render_table(measurements: list[nmeasure.LabelMeasurement]) -> None:
    table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
    table.add_column("structure")
    for column in ("volume (mL)", "max Ø (mm)", "surface (cm²)", "sphericity", "parts"):
        table.add_column(column, justify="right")
    for m in measurements:
        table.add_row(
            m.name + (" *" if m.resolution_limited else ""),
            f"{m.volume_ml:.2f}", f"{m.max_diameter_mm:.1f}",
            f"{m.surface_area_mm2 / 100:.1f}", f"{m.sphericity:.3f}", str(m.n_components),
        )
    console.print(table)
    if any(m.resolution_limited for m in measurements):
        console.print(
            f"[dim]* spans under {nmeasure.RESOLUTION_LIMIT_VOXELS:.0f} voxels on its "
            "thinnest axis; shape figures are indicative only[/dim]"
        )


@app.command()
def measure(
    labelmap: Path = typer.Argument(..., help="Segmentation volume, or a DICOM directory."),
    labels: str = typer.Option(None, "--labels", "-l", help="Names, e.g. '1=liver,2=tumour'."),
    split: str = typer.Option(None, "--split", help="Labels to split into components, e.g. '2'."),
    min_volume: float = typer.Option(0.0, "--min-volume", help="Drop components below N mm³."),
    json_out: Path = typer.Option(None, "--json", help="Also write measurements as JSON."),
) -> None:
    """Report physical measurements for every label in a segmentation."""
    volume = nio.load(labelmap)
    names = _parse_labels(labels)
    split_labels = _parse_split(split)

    results: list[nmeasure.LabelMeasurement] = []
    for m in nmeasure.measure_all(volume, names):
        if m.label in split_labels:
            results.extend(
                nmeasure.measure_components(volume, m.label, m.name, min_volume_mm3=min_volume)
            )
        else:
            results.append(m)

    console.print(f"[bold]{volume.name}[/bold]  {volume!r}")
    _render_table(results)

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps([m.as_dict() for m in results], indent=2))
        console.print(f"[dim]wrote {json_out}[/dim]")


@app.command()
def view(
    labelmap: Path = typer.Argument(..., help="Segmentation volume to render."),
    output: Path = typer.Option("scene.html", "--output", "-o", help="Destination HTML file."),
    labels: str = typer.Option(None, "--labels", "-l", help="Names, e.g. '1=liver,2=tumour'."),
    split: str = typer.Option(None, "--split", help="Labels to split into components."),
    max_faces: int = typer.Option(60_000, "--max-faces", help="Triangle budget per structure."),
    smooth: int = typer.Option(8, "--smooth", help="Taubin smoothing iterations."),
    min_volume: float = typer.Option(0.0, "--min-volume", help="Drop components below N mm³."),
    title: str = typer.Option(None, "--title", help="Scene title."),
) -> None:
    """Build an interactive 3-D viewer from any labelled segmentation."""
    volume = nio.load(labelmap)
    scene = scene_from_labelmap(
        volume,
        names=_parse_labels(labels),
        split_labels=_parse_split(split),
        max_faces_per_structure=max_faces,
        smooth_iterations=smooth,
        min_component_volume_mm3=min_volume,
        title=title or volume.name,
        subtitle=str(labelmap),
    )
    path = nhtml.write(scene, output)
    size_kb = path.stat().st_size / 1024
    console.print(
        f"[green]wrote[/green] {path}  "
        f"[dim]({len(scene.meshes)} structures, {scene.total_faces():,} triangles, "
        f"{size_kb:,.0f} KB)[/dim]"
    )
    _render_table(scene.measurements)


@app.command()
def prep(
    image: Path = typer.Argument(..., help="CT volume or DICOM directory, in Hounsfield units."),
    output: Path = typer.Argument(..., help="Destination volume, e.g. out.nrrd."),
    window: str = typer.Option("abdomen", "--window", "-w", help=f"One of {sorted(HU_WINDOWS)}."),
    isotropic: float = typer.Option(1.0, "--isotropic", help="Target mm/voxel; 0 to skip."),
    denoise_mm: float = typer.Option(0.0, "--denoise", help="Denoise width in mm; 0 to skip."),
    keep_table: bool = typer.Option(False, "--keep-table", help="Do not strip the scanner table."),
) -> None:
    """Preprocess a CT volume and write the result."""
    volume = nio.load(image)
    config = PreprocessConfig(
        window=window or None,
        isotropic_mm=isotropic or None,
        denoise_mm=denoise_mm,
        strip_table=not keep_table,
    )
    console.print(f"[dim]{config.describe()}[/dim]")
    console.print(f"  in  {volume!r}")
    result = preprocess_run(volume, config)
    console.print(f"  out {result!r}")
    console.print(f"[green]wrote[/green] {nio.save(result, output)}")


@app.command()
def convert(
    source: Path = typer.Argument(..., help="Input volume or DICOM directory."),
    output: Path = typer.Argument(..., help="Output volume; format follows the extension."),
) -> None:
    """Convert between volume formats, preserving spacing and origin."""
    volume = nio.load(source)
    console.print(f"{volume!r}")
    console.print(f"[green]wrote[/green] {nio.save(volume, output)}")


@app.command()
def info(source: Path = typer.Argument(..., help="Volume or DICOM directory.")) -> None:
    """Print geometry and intensity statistics for a volume."""
    volume = nio.load(source)
    console.print(f"[bold]{volume.name}[/bold]")
    rows = {
        "grid (z, y, x)": " x ".join(str(n) for n in volume.shape),
        "spacing (mm)": " x ".join(f"{s:.4g}" for s in volume.spacing),
        "field of view (mm)": " x ".join(f"{e:.1f}" for e in volume.extent_mm),
        "voxel volume (mm³)": f"{volume.voxel_volume_mm3:.5g}",
        "isotropic": "yes" if volume.is_isotropic else "no",
        "dtype": str(volume.array.dtype),
        "intensity range": f"{volume.array.min():.6g} to {volume.array.max():.6g}",
    }
    for key, value in rows.items():
        console.print(f"  {key:<20} {value}")


@app.command()
def demo(
    output: Path = typer.Option("outputs/demo.html", "--output", "-o", help="Destination HTML."),
    lesions: int = typer.Option(3, "--lesions", help="Lesions to place in the phantom."),
) -> None:
    """Run the whole pipeline on a synthetic phantom, no download required."""
    from .phantom import TORSO_LABEL_NAMES, torso_phantom

    _, labelmap = torso_phantom(n_lesions=lesions)
    scene = scene_from_labelmap(
        labelmap, names=TORSO_LABEL_NAMES, split_labels=(2,),
        title="Synthetic torso phantom",
        subtitle="Generated by voxelmetry.phantom - no patient data",
    )
    path = nhtml.write(scene, output)
    console.print(f"[green]wrote[/green] {path} [dim]({path.stat().st_size / 1024:,.0f} KB)[/dim]")
    _render_table(scene.measurements)


if __name__ == "__main__":  # pragma: no cover
    app()
