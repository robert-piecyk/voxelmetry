"""Measure every case in a dataset and summarise the cohort.

Works on any Medical Segmentation Decathlon task without modification: label
names come from ``dataset.json``, so the same script profiles liver, spleen or
pancreas. Output is one JSON record per case plus a printed summary.

    python examples/cohort_report.py /path/to/Task03_Liver \\
        --organ 1 --lesion 2 --workers 16 --out outputs/liver_cohort.json

Cases whose geometry disagrees with the rest of the cohort usually indicate
something about the acquisition or the annotation rather than the anatomy.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from voxelmetry.datasets import MSDDataset
from voxelmetry.io import load
from voxelmetry.measure import lesion_burden, measure_components, measure_label


def profile_case(args: tuple[str, str, int, int | None, float, bool]) -> dict:
    """Measure one case. Returns a record, or an ``error`` key on failure.

    Runs in a worker process, so it takes plain arguments and returns plain
    data rather than Volume objects.
    """
    root, case_id, organ_label, lesion_label, min_lesion_mm3, with_intensity = args
    try:
        dataset = MSDDataset(root)
        case = dataset.case(case_id)
        # The label volume is all the morphometry needs. Decompressing the
        # matching image as well roughly doubles the I/O for two numbers, so
        # it is opt-in.
        labels = load(case.label_path) if case.label_path else None
        if labels is None:
            return {"case_id": case_id, "error": "no label volume"}

        organ = measure_label(labels, organ_label, "organ")
        record = {
            "case_id": case_id,
            "grid": list(labels.shape),
            "spacing_mm": list(labels.spacing),
            "voxel_volume_mm3": labels.voxel_volume_mm3,
            "extent_mm": list(labels.extent_mm),
            "slice_thickness_mm": labels.spacing[0],
            "organ": organ.as_dict(),
        }
        if with_intensity:
            image = load(case.image_path)
            record["hu_min"] = float(image.array.min())
            record["hu_max"] = float(image.array.max())

        if lesion_label is not None:
            lesions = measure_components(
                labels, lesion_label, "lesion", min_volume_mm3=min_lesion_mm3
            )
            record["lesions"] = [m.as_dict() for m in lesions]
            record["burden"] = lesion_burden(lesions, reference_volume_ml=organ.volume_ml)
        return record
    except Exception:  # noqa: BLE001 - one bad case must not kill the run
        return {"case_id": case_id, "error": traceback.format_exc(limit=3)}


def read_records(path: Path | None) -> list[dict]:
    """Load records already written to a JSONL file, tolerating a partial line.

    A run killed mid-write can leave the final line truncated; that line is
    dropped rather than aborting the resume.
    """
    if path is None or not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def summarise(records: list[dict], label_names: dict[int, str]) -> str:
    """Render a text summary of the cohort, flagging outliers."""
    good = [r for r in records if "error" not in r]
    failed = [r for r in records if "error" in r]
    if not good:
        return "No cases could be measured."

    lines: list[str] = []
    add = lines.append

    add(f"cases measured        {len(good)}" + (f"  ({len(failed)} failed)" if failed else ""))

    thickness = np.array([r["slice_thickness_mm"] for r in good])
    in_plane = np.array([r["spacing_mm"][1] for r in good])
    add(f"slice thickness (mm)  {thickness.min():.2f} to {thickness.max():.2f}"
        f"   median {np.median(thickness):.2f}")
    add(f"in-plane spacing (mm) {in_plane.min():.3f} to {in_plane.max():.3f}"
        f"  median {np.median(in_plane):.3f}")
    add(f"anisotropy (z / x)    {np.median(thickness / in_plane):.1f}x median,"
        f" up to {(thickness / in_plane).max():.1f}x")

    volumes = np.array([r["organ"]["volume_ml"] for r in good])
    add("")
    add(f"organ volume (mL)     {volumes.min():.0f} to {volumes.max():.0f}"
        f"   median {np.median(volumes):.0f}   IQR "
        f"{np.percentile(volumes, 25):.0f}-{np.percentile(volumes, 75):.0f}")

    parts = np.array([r["organ"]["n_components"] for r in good])
    fragmented = [r for r in good if r["organ"]["n_components"] > 1]
    add(f"organ components      {int(parts.min())} to {int(parts.max())}"
        f"   {len(fragmented)} case(s) not a single connected region")

    # Component counts at face-adjacency are dominated by single voxels touching
    # the main body at a corner, so a label can report hundreds of parts for a
    # few hundredths of a percent of its volume. Stray volume is the usable
    # figure.
    stray = np.array([
        r["organ"]["volume_mm3"] * (1.0 - r["organ"]["largest_component_fraction"])
        for r in good
    ])
    add(f"  stray volume (mm3)  {stray.min():.1f} to {stray.max():.1f}"
        f"   median {np.median(stray):.1f}"
        f"   worst {100 * (stray / np.array([r['organ']['volume_mm3'] for r in good])).max():.3f}%"
        " of its organ")

    if any("burden" in r for r in good):
        with_lesions = [r for r in good if r.get("burden", {}).get("n_lesions", 0) > 0]
        add("")
        add(f"cases with lesions    {len(with_lesions)} of {len(good)}")
        if with_lesions:
            counts = np.array([r["burden"]["n_lesions"] for r in with_lesions])
            burden = np.array([r["burden"].get("burden_percent", 0.0) for r in with_lesions])
            largest = np.array([r["burden"]["largest_diameter_mm"] for r in with_lesions])
            add(f"lesions per case      {counts.min()} to {counts.max()}"
                f"   median {int(np.median(counts))}   total {int(counts.sum())}")
            add(f"lesion burden (%)     {burden.min():.2f} to {burden.max():.1f}"
                f"   median {np.median(burden):.2f}")
            add(f"largest lesion Ø (mm) {largest.min():.0f} to {largest.max():.0f}"
                f"   median {np.median(largest):.0f}")

    # Outliers: cases whose organ volume sits far from the cohort centre.
    # Median absolute deviation rather than standard deviation, so a couple of
    # extreme cases do not inflate the threshold and hide themselves.
    median = np.median(volumes)
    mad = np.median(np.abs(volumes - median)) or 1.0
    z = 0.6745 * (volumes - median) / mad
    flagged = [(good[i], z[i]) for i in np.argsort(-np.abs(z))[:5] if abs(z[i]) > 3.5]
    if flagged:
        add("")
        add("volume outliers (robust z > 3.5):")
        for record, score in flagged:
            add(f"  {record['case_id']:<16} {record['organ']['volume_ml']:>8.0f} mL"
                f"   z={score:+.1f}   thickness {record['slice_thickness_mm']:.2f} mm")

    if fragmented:
        add("")
        add("organ segmented as more than one region"
            " (components at face-adjacency; 26-adjacency gives far fewer):")
        for record in sorted(fragmented, key=lambda r: -r["organ"]["n_components"])[:8]:
            organ = record["organ"]
            # A fraction that rounds to 100% cannot distinguish a genuine
            # second lobe from a handful of mislabelled voxels; the absolute
            # stray volume can.
            stray_mm3 = organ["volume_mm3"] * (1.0 - organ["largest_component_fraction"])
            add(f"  {record['case_id']:<16} {organ['n_components']:>3} parts,"
                f" {stray_mm3:>8.1f} mm3 outside the main region"
                f"  ({stray_mm3 / record['voxel_volume_mm3']:.0f} voxels)")

    if failed:
        add("")
        add("failed cases:")
        for record in failed[:5]:
            first_line = str(record["error"]).strip().splitlines()[-1][:90]
            add(f"  {record['case_id']:<16} {first_line}")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Unpacked MSD task directory.")
    parser.add_argument("--organ", type=int, default=1, help="Label value of the organ.")
    parser.add_argument("--lesion", type=int, default=None, help="Label value of lesions.")
    parser.add_argument("--min-lesion-mm3", type=float, default=0.0,
                        help="Ignore lesion components below this volume.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker processes.")
    parser.add_argument("--limit", type=int, default=None, help="Measure only the first N cases.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Stream per-case records to this JSONL file, resuming if it exists.")
    parser.add_argument("--with-intensity", action="store_true",
                        help="Also load the image volume to record its HU range (slower).")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Summarise an existing --out file without measuring anything.")
    args = parser.parse_args(argv)

    dataset = MSDDataset(args.root)
    cases = dataset.cases[: args.limit]
    print(f"{dataset.name}: {len(cases)} cases, labels {dataset.label_names}", file=sys.stderr)

    # Results stream to JSONL as they land, so an interrupted run over 131 large
    # volumes keeps what it measured and resumes rather than restarting.
    records = read_records(args.out) if args.out else []
    done_ids = {r["case_id"] for r in records if "error" not in r}
    if done_ids:
        print(f"resuming: {len(done_ids)} already measured", file=sys.stderr)

    todo = [c for c in cases if c.case_id not in done_ids]
    if args.summarize_only:
        todo = []

    if todo:
        payloads = [
            (str(args.root), c.case_id, args.organ, args.lesion,
             args.min_lesion_mm3, args.with_intensity)
            for c in todo
        ]
        handle = None
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            handle = args.out.open("a")
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(profile_case, p): p[1] for p in payloads}
                for done, future in enumerate(as_completed(futures), start=1):
                    record = future.result()
                    records.append(record)
                    if handle:
                        handle.write(json.dumps(record) + "\n")
                        handle.flush()
                    print(f"\r  {done}/{len(futures)}", end="", file=sys.stderr, flush=True)
        finally:
            if handle:
                handle.close()
        print(file=sys.stderr)

    records.sort(key=lambda r: r["case_id"])
    if args.out:
        print(f"{len(records)} records in {args.out}", file=sys.stderr)

    print()
    print(summarise(records, dataset.label_names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
