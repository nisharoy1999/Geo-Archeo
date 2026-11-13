#!/usr/bin/env bash
# run_background_builds.sh
#
# Builds the background-design conditions for pipeline/exp_backgrounds.py. Run it from the
# GeoArcheo folder on the Mac, after download_seasons.py and download_fabdem.py, i.e. when
# data/raw_seasons/ and data/raw_fabdem/ exist. Each build re-reads every tile for one
# background draw, so this takes a while: start with ONE condition and ONE seed to time it.
#
#   bash run_background_builds.sh                 # all conditions, 5 draws each
#   SEEDS="123 7" bash run_background_builds.sh   # fewer draws (quicker; results less stable)
#   ONLY=km5 bash run_background_builds.sh        # one condition
#
# Files already built are skipped, so it can be stopped and restarted. If a build was interrupted,
# delete its half-written file before restarting.
#
#   conditions                    meaning
#   km5, km10                     random backgrounds at least 5 km / 10 km from every known site
#   matched_km2.5, matched_km5    backgrounds with the same whole-patch terrain roughness and slope
#                                 as the site (chosen from 40 random candidates in the same tile)
#
# The paper's own condition ("main": random, at least 2.5 km) is the data you already have.

set -e
SEEDS="${SEEDS:-123 7 11 19 42}"
DEM="--dem-dir data/raw_fabdem --extra-tag _fabdem"

build() {
  name="$1"; shift
  if [ -n "$ONLY" ] && [ "$ONLY" != "$name" ]; then return; fi
  mkdir -p "data/bg/$name"
  for s in $SEEDS; do
    if [ "$s" = "123" ]; then out="data/bg/$name/season_patches.npz"; else out="data/bg/$name/season_patches_s$s.npz"; fi
    if [ -f "$out" ]; then echo "skip $out (exists)"; continue; fi
    echo "=== $name, background seed $s ==="
    python build_season_patches.py --seed "$s" --out "$out" $DEM "$@"
  done
}

build km5 --min-km 5
build km10 --min-km 10
build matched_km2.5 --terrain-matched --min-km 2.5
build matched_km5 --terrain-matched --min-km 5
echo "Done. Now: python -m pipeline.exp_backgrounds"
