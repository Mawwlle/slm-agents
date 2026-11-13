---
name: shell-data-tasks
description: How to solve small file and data tasks in a shell and report the answer.
---
1. Look before acting: `ls -R`, `head -n 5 <file>`, `wc -l <file>`.
2. Prefer one precise command over many: grep -c, sort | uniq -c | sort -rn, awk -F, for sums, or a short `python3 -c` for anything with CSV/JSON/dates.
3. Check your result once with a different command (e.g. count with grep and with python) before you answer.
4. When a file must be written, write exactly the requested format (one item per line, valid JSON, no extra text) and `cat` it to check.
5. To fix code, run its tests first, read the error, change the code (never the tests), run the tests again.
6. Call finish with only the value in `answer` (a number or a path), no sentence around it.
