"""Export brainscope directions as hotwire vector files — with a passport.

Writes one ``<name>.pt`` per direction (the tensor hotwire's
``$HOTWIRE_VECTORS`` loader expects: ``(hidden,)`` or ``(n_layers, hidden)``)
plus a ``manifest.json`` recording what each vector IS: source, shape, norms,
the model it was extracted for, and — the field that exists because its
absence once cost a debugging day — the steering **regime** it was calibrated
in. A vector calibrated on generation-only steering must ship with
``decode_only: true``; applying it to a long prompt as well multiplies the
effective dose and can wreck coherence.

    python -m brainscope.export_hotwire --dirs dirs.json --out ./vectors
    python -m brainscope.export_hotwire --dirs dirs.json --out ./vectors \
        --names calm,formal --model Qwen/Qwen3-4B-Instruct-2507
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


def export(dirs_path: str, out_dir: str, names: list[str] | None = None,
           model: str | None = None, meta_path: str | None = None) -> dict:
    """Export directions to hotwire .pt files + manifest. Returns the manifest."""
    dirs = json.loads(Path(dirs_path).read_text())
    meta = {}
    if meta_path and Path(meta_path).exists():
        meta = json.loads(Path(meta_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    selected = names if names else list(dirs)
    entries = []
    for name in selected:
        if name not in dirs:
            raise KeyError(f"direction {name!r} not in {dirs_path}")
        t = torch.tensor(dirs[name], dtype=torch.float32)
        safe = name.replace("/", "_").replace(":", "_")
        torch.save(t, out / f"{safe}.pt")
        norms = ([round(float(r.norm()), 4) for r in t] if t.dim() == 2
                 else [round(float(t.norm()), 4)])
        entries.append({
            "name": name,
            "file": f"{safe}.pt",
            "shape": list(t.shape),
            "norm_min": min(norms),
            "norm_max": max(norms),
            "unit_norm": max(norms) <= 1.001 and min(norms) >= 0.999,
            "model": model or meta.get("model"),
            "source": str(dirs_path),
            "calibration_regime": {
                "backend": "brainscope",
                "note": ("brainscope steers generation (and mutes tool-call "
                         "JSON scaffolding); it steers the prompt only in "
                         "free-text requests. When deploying via hotwire, "
                         "start with decode_only: true unless the vector was "
                         "explicitly calibrated with prompt steering."),
                "decode_only_recommended": True,
            },
        })

    manifest = {"exported_at": datetime.now(timezone.utc).isoformat(),
                "vectors": entries}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dirs", required=True, help="dirs.json path")
    ap.add_argument("--meta", default=None, help="dirs.meta.json path (optional)")
    ap.add_argument("--out", required=True, help="output directory (HOTWIRE_VECTORS)")
    ap.add_argument("--names", default=None,
                    help="comma-separated direction names (default: all)")
    ap.add_argument("--model", default=None, help="model id for the passport")
    args = ap.parse_args(argv)
    names = [n.strip() for n in args.names.split(",")] if args.names else None
    manifest = export(args.dirs, args.out, names, args.model, args.meta)
    for e in manifest["vectors"]:
        tag = "unit-norm" if e["unit_norm"] else f"norm {e['norm_min']}..{e['norm_max']}"
        print(f"exported {e['name']:<40} {e['shape']} ({tag})")
    print(f"manifest -> {Path(args.out) / 'manifest.json'}")


if __name__ == "__main__":
    main()
