"""Stage one immutable dump copy shared only by this experiment's jobs."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import time

from .common import write_json


def stage_dump(source, target):
    source, target = Path(source).resolve(), Path(target)
    target.mkdir(parents=True, exist_ok=True)
    with (target / "stage.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        files = ("layer_33.npy", "rows_tok.npy", "norm_params_layer33.npz", "meta.json")
        for name in files:
            src, dst = source/name, target/name
            stat = src.stat()
            identity = {"path": str(src), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            receipt = target/(name+".receipt.json")
            if dst.exists() and receipt.exists():
                import json
                previous = json.loads(receipt.read_text())
                if previous["source"] != identity or dst.stat().st_size != stat.st_size:
                    raise ValueError(f"Immutable stage/source mismatch: {dst}")
                print(f"reuse staged {name}", flush=True)
                continue
            free = shutil.disk_usage(target).free
            if free < stat.st_size + 4*1024**3:
                raise RuntimeError(f"Not enough free staging space: need {stat.st_size}, have {free}")
            temporary = target/(name+".partial")
            digest = hashlib.sha256(); start = time.monotonic(); done = 0; progress = start
            with src.open("rb") as reader, temporary.open("wb") as writer:
                while data := reader.read(32*1024**2):
                    writer.write(data); digest.update(data); done += len(data)
                    if time.monotonic()-progress > 30:
                        print(f"stage {name}: {done/stat.st_size:.1%}", flush=True); progress = time.monotonic()
            end_stat = src.stat()
            if (end_stat.st_size, end_stat.st_mtime_ns) != (stat.st_size, stat.st_mtime_ns):
                temporary.unlink(); raise ValueError("Source changed during staging")
            temporary.replace(dst); os.chmod(dst, 0o444)
            write_json(receipt, {"source": identity, "sha256": digest.hexdigest(),
                                "seconds": time.monotonic()-start})
        return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True); parser.add_argument("--target", required=True)
    args = parser.parse_args(); stage_dump(args.source, args.target)


if __name__ == "__main__": main()
