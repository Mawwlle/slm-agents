"""Download the episode logs listed in data/episodes.json into data/episodes/ and check their sha256.

    python data/fetch.py            # all files (954 MB)
    python data/fetch.py stage 4    # only files whose stage starts with "stage 4"

Files already present with the right hash are skipped.
"""

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

DATASET = 'https://huggingface.co/datasets/<org>/<name>/resolve/main'  # set when the dataset is published
HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    wanted = ' '.join(sys.argv[1:])
    manifest = json.loads((HERE / 'episodes.json').read_text())
    for name, entry in manifest.items():
        if not entry['stage'].startswith(wanted):
            continue
        target = HERE / 'episodes' / name
        if target.exists() and sha256(target) == entry['sha256']:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f'{DATASET}/{name}', target)
        if sha256(target) != entry['sha256']:
            sys.exit(f'{name}: sha256 mismatch')
        print(f"{name}  {entry['episodes']} episodes")


if __name__ == '__main__':
    main()
