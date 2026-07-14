#!/usr/bin/env bash
# Prune stale temp that accumulates in the RAM-backed /tmp tmpfs on this box and
# can starve a full benchmark run's index build (see .env TMPDIR note).
#
# Two sources, both pruned PID-safely (never touch a dir owned by a live PID):
#   1. ram-diskann-* : leaked temp dirs from a DiskANN Rust test suite run by
#      khora-side work. NOT created by this benchmark (its venv has no LanceDB),
#      but they share /tmp, so a leak here can still wedge a run.
#   2. leftover khora index scratch older than the age threshold.
#
# Safe to run any time - it skips anything a running process still owns. Wire a
# cron/systemd-timer to it if the box does frequent khora dev + benchmark runs.
#
# Usage: scripts/prune-tmp.sh [MAX_AGE_MINUTES]   (default 120)
set -uo pipefail

TMPROOT="${TMPDIR_SCAN:-/tmp}"
MAX_AGE_MIN="${1:-120}"

running=" $(ps -eo pid= | tr -s '\n ' '  ') "

# Extract the PID embedded in a "ram-diskann-<magic>-<test>-<PID>-<nonce>" name.
_owner_pid() {
    local base="${1##*/}" no_nonce
    no_nonce="${base%-*}"   # drop trailing -<nonce>
    printf '%s' "${no_nonce##*-}"
}

deleted=0
kept=0
while IFS= read -r -d '' d; do
    pid="$(_owner_pid "$d")"
    case "$running" in
        *" $pid "*) kept=$((kept + 1)); continue ;;
    esac
    rm -rf "$d" && deleted=$((deleted + 1))
done < <(find "$TMPROOT" -maxdepth 1 -name 'ram-diskann-*' -mmin "+${MAX_AGE_MIN}" -print0 2>/dev/null)

echo "prune-tmp: removed ${deleted} stale ram-diskann dir(s) (> ${MAX_AGE_MIN} min old), kept ${kept} owned by live PIDs"
df -h "$TMPROOT" 2>/dev/null | awk 'NR==2{print "prune-tmp: " $6 " now " $3 " used, " $4 " free (" $5 ")"}'
