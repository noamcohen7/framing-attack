#!/usr/bin/env bash
# Build the course submission tarball:
#   <groupname>.tar.gz  ->  <groupname>/code/   (committed code + README.md)
#                           <groupname>/report/ (LaTeX source + PDF)
#
# Usage: scripts/make_submission.sh <groupname> <path/to/report.pdf>
#
# Code is taken from the last commit with `git archive`, so untracked and
# git-ignored files (.env with API keys, results/, .venv/) are never included.
set -euo pipefail

group="${1:?usage: $0 <groupname> <report.pdf>}"
pdf="${2:?usage: $0 <groupname> <report.pdf>}"
[ -f "$pdf" ] || { echo "PDF not found: $pdf" >&2; exit 1; }

root="$(git rev-parse --show-toplevel)"
cd "$root"
if ! git diff --quiet HEAD -- . ; then
  echo "Uncommitted changes: commit first so the tarball matches the repository." >&2
  exit 1
fi

out="$(mktemp -d)"
mkdir -p "$out/$group/code" "$out/$group/report"
git archive HEAD | tar -x -C "$out/$group/code"
# Personal / session notes are not part of the submission.
rm -f "$out/$group/code/my_text_for_claude" "$out/$group/code/NEXT_SESSION.md"
cp paper/final_report.tex "$out/$group/report/"
cp "$pdf" "$out/$group/report/${group}_report.pdf"

# Safety check: no secrets in the package.
if find "$out/$group" -name ".env" -o -name "*.key" | grep -q . ; then
  echo "Refusing: a secrets file ended up in the package." >&2; exit 1
fi

tar -czf "$root/$group.tar.gz" -C "$out" "$group"
rm -rf "$out"
echo "Wrote $root/$group.tar.gz"
tar -tzf "$root/$group.tar.gz" | awk -F/ 'NF<=3' | sort -u | head -40
