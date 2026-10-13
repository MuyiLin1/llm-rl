"""Rebuild export/ from the repo: copy (into export/repo/) every code / log / figure / doc / LaTeX file,
skip bulk binary data (listed in DATA_MANIFEST.tsv instead), and pull files that are
deleted from the working tree but still in git HEAD into _from_git_head/.

Run from anywhere:  python3 export/_build_export.py
It wipes and rebuilds everything in export/ except this script and README.md.
"""
import os, shutil, subprocess, sys, zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "export"
KEEP = {"_build_export.py", "README.md"}

DATA_EXT = {".gz", ".npz", ".npy", ".pkl", ".pickle", ".zip"}
JUNK_NAMES = {".DS_Store"}
JUNK_EXT = {".pyc", ".pid"}
SKIP_DIRS = {".git", "__pycache__", "export"}
MAX_BYTES = 50e6  # anything bigger is treated as data (only hit by the Miniconda installer)


def npz_summary(p):
    """Array names, shapes and dtypes from the .npy headers, without loading the data."""
    try:
        if p.suffix == ".npy":
            a = np.load(p, mmap_mode="r")
            return f"{a.shape} {a.dtype}"
        parts = []
        with zipfile.ZipFile(p) as z:
            for name in z.namelist():
                with z.open(name) as f:
                    v = np.lib.format.read_magic(f)
                    read = (np.lib.format.read_array_header_1_0 if v == (1, 0)
                            else np.lib.format.read_array_header_2_0)
                    shape, _, dtype = read(f)
                parts.append(f"{name[:-4]}{list(shape)}:{dtype}")
        return "; ".join(parts)
    except Exception as e:
        return f"(unreadable: {e.__class__.__name__})"


def main():
    for child in OUT.iterdir():
        if child.name in KEEP:
            continue
        shutil.rmtree(child) if child.is_dir() else child.unlink()

    copied, data_rows, other_skipped = [], [], []
    for root, dirs, files in os.walk(REPO, followlinks=False):
        rel_root = Path(root).relative_to(REPO)
        if rel_root == Path("."):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        else:
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
        for fn in files:
            src = Path(root) / fn
            rel = src.relative_to(REPO)
            if src.is_symlink():
                other_skipped.append((rel, "symlink -> " + os.readlink(src)))
                continue
            size = src.stat().st_size
            ext = src.suffix.lower()
            if fn in JUNK_NAMES or ext in JUNK_EXT:
                continue
            if ext in DATA_EXT or size > MAX_BYTES:
                info = npz_summary(src) if ext in {".npz", ".npy"} else ""
                data_rows.append((str(rel), size, info))
                continue
            dst = OUT / "repo" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append((str(rel), size))

    # Files tracked in HEAD but deleted from the working tree.
    deleted = subprocess.run(["git", "-C", str(REPO), "ls-files", "-d"],
                             capture_output=True, text=True, check=True).stdout.split()
    for rel in deleted:
        blob = subprocess.run(["git", "-C", str(REPO), "show", f"HEAD:{rel}"],
                              capture_output=True, check=True).stdout
        dst = OUT / "_from_git_head" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(blob)

    meta = OUT / "_meta"
    meta.mkdir()
    git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a],
                                    capture_output=True, text=True).stdout
    (meta / "git_log.txt").write_text(git("log", "--stat", "--date=iso"))
    (meta / "git_status.txt").write_text(git("status", "--short", "--untracked-files=no"))
    (meta / "git_uncommitted_changes.diff").write_text(git("diff", "--", "amazoncat13k"))

    with open(meta / "DATA_MANIFEST.tsv", "w") as f:
        f.write("path\tbytes\tarrays (npz/npy only)\n")
        for rel, size, info in sorted(data_rows):
            f.write(f"{rel}\t{size}\t{info}\n")
    with open(meta / "FILES_INCLUDED.tsv", "w") as f:
        f.write("path\tbytes\n")
        for rel, size in sorted(copied):
            f.write(f"{rel}\t{size}\n")
    for rel, why in other_skipped:
        print(f"skipped {rel}: {why}")

    tot = sum(s for _, s in copied)
    dtot = sum(s for _, s, _ in data_rows)
    print(f"copied {len(copied)} files ({tot/1e6:.1f} MB); "
          f"{len(deleted)} from git HEAD; "
          f"{len(data_rows)} data files listed, not copied ({dtot/1e9:.1f} GB)")


if __name__ == "__main__":
    sys.exit(main())
