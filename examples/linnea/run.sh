#!/usr/bin/env bash
# Linnea (https://github.com/HPAC/linnea, last changed 2021) on the problems
# of ulm.lam's examples: its best algorithm for each, kept in generated/.
# It needs an older parser library and pkg_resources: a virtual environment
# with Python 3.12, TatSu 5.6.1 and setuptools < 70 (in build/linnea).
# (Linnea's --time-limit option fails; the default settings are used.)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HERE/../.."
W="$ROOT/build/linnea"
mkdir -p "$W"
if [ ! -x "$W/venv/bin/linnea" ]; then
	[ -d "$W/src" ] || git clone --quiet https://github.com/HPAC/linnea.git "$W/src"
	python3.12 -m venv "$W/venv"
	"$W/venv/bin/pip" install --quiet -e "$W/src" "tatsu==5.6.1" "setuptools<70"
fi
cd "$W"
for f in "$HERE"/*.la; do
	n=$(basename "$f" .la)
	rm -rf "$n"
	"$W/venv/bin/linnea" -o "$n" "$f" . >"$n.log" 2>&1
	cp "$n/Julia/generated/algorithm0.jl" "$HERE/generated/$n.jl"
	printf '%-14s best solution %s FLOPs\n' "$n" "$(grep 'Best solution' "$n.log" | awk '{print $3}')"
done
