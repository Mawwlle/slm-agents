"""The project page's data → docs/data.js, from the same results and macros as the paper; and the PDF → docs/paper.pdf.

    python -m paper.site        (after paper.macros and the paper build)

data.js sets window.DATA rather than being fetched as JSON, so the page also opens from file://.
"""

import json
import re
import shutil

from analysis.data import ROOT
from paper.macros import load, numbers
from paper.tables import CLAIMS

DOCS = ROOT / 'docs'
SYMBOLS = {'{,}': ',', '\\%': '%', '\\,': '\u2009', '\\le': '≤', '\\cap': '∩', '$': ''}


def plain(tex: str) -> str:
    """The paper's LaTeX as page text: 221{,}493 → 221,493, 66\\% → 66%, $-15.9$ → −15.9, $\\le$ → ≤."""

    for symbol, text in SYMBOLS.items():
        tex = tex.replace(symbol, text)
    return re.sub(r'(?<!\w)-(?=\d)', '−', tex)


def main() -> None:
    verdicts = load('verdicts')
    data = {
        'numbers': {k: plain(v) for k, v in numbers().items()},
        'replace': load('c1'),
        'delegate': load('c4'),
        'cheap': load('c5'),
        'verdicts': [{'id': k, 'claim': plain(claim), 'holds': verdicts[k]['verdict'] == 'HOLDS', 'p': verdicts[k]['p']}
                     for k, claim in CLAIMS.items()],
    }
    (DOCS / 'data.js').write_text('// written by paper/site.py — do not edit\nwindow.DATA = ' + json.dumps(data) + ';\n')
    shutil.copy(ROOT / 'paper/build/main.pdf', DOCS / 'paper.pdf')
    print('docs/data.js, docs/paper.pdf')


if __name__ == '__main__':
    main()
