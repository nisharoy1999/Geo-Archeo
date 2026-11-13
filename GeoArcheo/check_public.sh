#!/usr/bin/env bash
# Run before every push: refuses if anything sensitive is tracked or staged.
set -e
bad=$(git ls-files | grep -E '\.(tif|tiff|npz|npy|pt|pth|geojson|gpkg|shp|pem|key)$|^data/|^outputs/|site_locations|eamena_|\.env$' || true)
if [ -n "$bad" ]; then echo "STOP: these tracked files must not be public:"; echo "$bad"; exit 1; fi
hits=$(git ls-files | grep -v '^check_public.sh$' | xargs grep -I -n -E 'csrftoken=|sessionid=|AKIA[0-9A-Z]{16}|BEGIN (RSA|OPENSSH) PRIVATE KEY' 2>/dev/null || true)
if [ -n "$hits" ]; then echo "STOP: possible secrets:"; echo "$hits"; exit 1; fi
big=$(git ls-files -z | xargs -0 du -k 2>/dev/null | awk '$1>5000{print}' || true)
if [ -n "$big" ]; then echo "WARNING: large files (KB):"; echo "$big"; fi
echo "OK: nothing sensitive is tracked."
