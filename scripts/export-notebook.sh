#!/usr/bin/env bash
# Export notebook/ulm-lam.jl to docs/index.html: a static page with all
# outputs (run here once), for GitHub Pages.  Needs PlutoSliderServer in
# the notebook environment.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
cp "$ROOT/notebook/ulm-lam.jl" "$TMP/"
# the notebook activates its own directory: give it the environment, with
# the path of UlmLAM made absolute
sed "s|\"../julia/UlmLAM\"|\"$ROOT/julia/UlmLAM\"|" "$ROOT/notebook/Project.toml" >"$TMP/Project.toml"
sed "s|\"../julia/UlmLAM\"|\"$ROOT/julia/UlmLAM\"|" "$ROOT/notebook/Manifest.toml" >"$TMP/Manifest.toml"
(cd "$TMP" && julia --project="$ROOT/notebook" -e 'using PlutoSliderServer
    PlutoSliderServer.export_directory("."; Export_baked_state=true,
        Export_offer_binder=false, Export_create_index=false)')
mkdir -p "$ROOT/docs"
cp "$TMP/ulm-lam.html" "$ROOT/docs/index.html"
touch "$ROOT/docs/.nojekyll"
echo "$ROOT/docs/index.html"
