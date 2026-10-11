"""Assemble the Zenodo data record from a clone of the original repository.

    python scripts/make_zenodo_bundle.py /path/to/original-repo --out /path/to/bundle

Every file of the original working tree is classified (see ``destination``):
code and small inputs moved into this repository, simulation maps and figures
moved to the Zenodo record, and a few files dropped (byte-code caches, an empty
log). Identical files stored several times are kept once.

Writes
------
``<out>/``                        the record: ``maps/<run>/``, ``figures/``,
                                  ``inputs/``, ``original/ReproducibleRun.zip``,
                                  ``README.md`` and ``MANIFEST.tsv``
                                  (path, size, sha256, original paths);
``provenance/ORIGINAL_INVENTORY.tsv``  every original file with its size,
                                  sha256 and destination (committed here).

Use ``--dry-run`` to write only the inventory.
"""

import argparse
import fnmatch
import hashlib
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from provenance.runs import RUNS, VACANCY_MAP  # noqa: E402

# Original path -> path in this repository.
TO_REPOSITORY = {
    "README.md": "legacy_version/README_original.md",
    "pyFi_adapt.py": "legacy_version/pyFi_adapt.py",
    "pyFi_RVE_cell.py": "legacy_version/pyFi_RVE_cell.py",
    "pyFi_origin_RVE_cell.ipynb": "legacy_version/pyFi_origin_RVE_cell.ipynb (outputs cleared)",
    "Post_Process.ipynb": "legacy_version/Post_Process.ipynb (outputs cleared)",
    "save_ipynb/main_RVE_cell.py": "legacy_version/save_ipynb/main_RVE_cell.py",
    "save_ipynb/pyFi_adapt.py": "legacy_version/save_ipynb/pyFi_adapt.py",
    "save_ipynb/pyFi_origin_RVE_cell.ipynb": "legacy_version/save_ipynb/pyFi_origin_RVE_cell.ipynb (outputs cleared)",
    "DSC_tool/compute_DSC.ipynb": "legacy_version/DSC_tool/compute_DSC.ipynb (outputs cleared)",
    "DSC_tool/free_nrj_mob.py": "legacy_version/DSC_tool/free_nrj_mob.py",
    "DSC_tool/3d02a_dsc_mean.py": "legacy_version/DSC_tool/3d02a_dsc_mean.py",
    "DSC_tool/3f01a_2dmap.py": "legacy_version/DSC_tool/3f01a_2dmap.py",
    "DSC_tool/heat capacity_data.csv": "data/experimental/heat_capacity_dsc.csv",
    "final_export.npz": "data/inputs/final_export.npz",
    "micro_final.npz": "data/inputs/micro_final.npz",
    "cyl_packing.npz": "data/inputs/cyl_packing.npz",
    "xyz_r.txt": "data/inputs/precipitates_xyz_r.txt",
    "xyz_r_big_cell.txt": "data/inputs/precipitates_xyz_r_big_cell.txt",
    "Preprint-2407.11580v1.pdf": "docs/Preprint-2407.11580v1.pdf",
    "save_fig/Supp_Mat_computed_microstructure_evolution.gif": "media/Supp_Mat_computed_microstructure_evolution.gif",
    "save_fig/Supp_Mat_computed_microstructure_evolution.mp4": "media/Supp_Mat_computed_microstructure_evolution.mp4",
    "ReproducibleRun.zip": "legacy_version/ReproducibleRun/ (code), data/reference/reproducible_run/ (first map, dtime.txt)",
}
# Small inputs are also copied to the record so that it is self-contained.
ALSO_IN_RECORD = {
    "final_export.npz": "inputs/final_export.npz",
    "micro_final.npz": "inputs/micro_final.npz",
    "cyl_packing.npz": "inputs/cyl_packing.npz",
    "xyz_r.txt": "inputs/precipitates_xyz_r.txt",
    "xyz_r_big_cell.txt": "inputs/precipitates_xyz_r_big_cell.txt",
    "ReproducibleRun.zip": "original/ReproducibleRun.zip",
    "save_fig/Supp_Mat_computed_microstructure_evolution.gif": "figures/Supp_Mat_computed_microstructure_evolution.gif",
    "save_fig/Supp_Mat_computed_microstructure_evolution.mp4": "figures/Supp_Mat_computed_microstructure_evolution.mp4",
}
DROPPED = {
    "*__pycache__/*": "Python byte-code cache",
    "log.txt": "17-byte stub log ('start simulation')",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_of(rel):
    for name, run in RUNS.items():
        if not run["maps"].startswith("ReproducibleRun/") and fnmatch.fnmatch(rel, run["maps"]):
            return name
    return None


def destination(rel, digest, run_digests):
    """Where an original file goes: ("repo", path), ("record", path) or ("dropped", reason)."""
    for pattern, reason in DROPPED.items():
        if fnmatch.fnmatch(rel, pattern):
            return [("dropped", reason)]
    out = []
    if rel in TO_REPOSITORY:
        out.append(("repo", TO_REPOSITORY[rel]))
    if rel in ALSO_IN_RECORD:
        out.append(("record", ALSO_IN_RECORD[rel]))
    if out:
        return out
    name = os.path.basename(rel)
    if rel == VACANCY_MAP:
        return [("record", f"maps/big_grid_1420/extra/{name[:-4]}_with_xv_out.npz")]
    if rel.endswith(".npz") and name.startswith("map"):
        # Maps stored outside their run folder (e.g. DSC_tool/) are identified by content.
        run = run_of(rel) or run_digests.get(digest)
        if run is not None:
            return [("record", f"maps/{run}/{name}")]
    if rel.startswith("save_fig/") or rel.endswith(".png"):
        return [("record", "figures/" + rel.removeprefix("save_fig/"))]
    raise ValueError(f"unclassified file: {rel}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("original", type=Path, help="clone of the original repository")
    parser.add_argument("--out", type=Path, help="bundle directory to create")
    parser.add_argument("--dry-run", action="store_true", help="only write provenance/ORIGINAL_INVENTORY.tsv")
    args = parser.parse_args(argv)
    if not args.dry_run and args.out is None:
        parser.error("--out is required unless --dry-run")

    files = sorted(
        str(p.relative_to(args.original)).replace(os.sep, "/")
        for p in args.original.rglob("*")
        if p.is_file() and ".git" not in p.relative_to(args.original).parts
    )
    digests = {rel: sha256(args.original / rel) for rel in files}
    run_digests = {digests[rel]: run_of(rel) for rel in files if run_of(rel)}

    inventory, record = [], {}
    for rel in files:
        dest = destination(rel, digests[rel], run_digests)
        inventory.append((rel, os.path.getsize(args.original / rel), digests[rel], dest))
        for kind, target in dest:
            if kind == "record":
                record.setdefault(target, (rel, digests[rel], []))[2].append(rel)

    # Maps of the reproducible run come from inside ReproducibleRun.zip.
    with zipfile.ZipFile(args.original / "ReproducibleRun.zip") as z:
        zip_maps = [n for n in z.namelist() if n.startswith("ReproducibleRun/saved_maps/map") and n.endswith(".npz")]

    inv_path = ROOT / "provenance" / "ORIGINAL_INVENTORY.tsv"
    with open(inv_path, "w", newline="\n") as f:
        f.write("original_path\tsize_bytes\tsha256\tdestination\n")
        for rel, size, digest, dest in inventory:
            f.write(f"{rel}\t{size}\t{digest}\t{'; '.join(f'{k}:{t}' for k, t in dest)}\n")
    print(f"wrote {inv_path.relative_to(ROOT)} ({len(inventory)} files)")
    if args.dry_run:
        return 0

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for target, (rel, digest, sources) in sorted(record.items()):
        dst = out / target
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.original / rel, dst)
        manifest.append((target, os.path.getsize(dst), digest, ";".join(sources)))
    with zipfile.ZipFile(args.original / "ReproducibleRun.zip") as z:
        for name in zip_maps:
            target = f"maps/reproducible_run/{os.path.basename(name)}"
            dst = out / target
            dst.parent.mkdir(parents=True, exist_ok=True)
            with z.open(name) as src, open(dst, "wb") as f:
                shutil.copyfileobj(src, f)
            manifest.append((target, os.path.getsize(dst), sha256(dst), "ReproducibleRun.zip:" + name))
    manifest.sort()
    with open(out / "MANIFEST.tsv", "w", newline="\n") as f:
        f.write("path\tsize_bytes\tsha256\toriginal_paths\n")
        for row in manifest:
            f.write("\t".join(str(x) for x in row) + "\n")
    shutil.copy(ROOT / "provenance" / "ZENODO_README.md", out / "README.md")
    total = sum(r[1] for r in manifest)
    print(f"bundle: {len(manifest)} files, {total / 1e9:.2f} GB in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
