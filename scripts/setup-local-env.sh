#!/usr/bin/env bash
#
# setup-local-env.sh — stamp the shared Bunny Storage config into this machine's .env.
#
# Why this exists: .env is gitignored AND Syncthing-ignored, so the Bunny vars
# don't travel between your machines — easy to forget on a fresh checkout and
# then wonder why local storage testing runs unconfigured. The Bunny secrets
# live in .env.bunny (gitignored but Syncthing-SYNCED, so it does travel). This
# script copies the BUNNY_* vars from .env.bunny into .env and forces
# DEPLOY_ENV=local, so local storage testing just works on any machine.
#
# Run once after cloning / syncing to a new machine (and any time .env.bunny changes):
#   scripts/setup-local-env.sh
#
# Idempotent: re-running updates the values in place.

set -euo pipefail
cd "$(dirname "$0")/.."

SOURCE="${SOURCE:-.env.bunny}"
TARGET="${TARGET:-.env}"

[ -f "$SOURCE" ] || { echo "ERROR: $SOURCE not found. It should arrive via Syncthing; check that the repo is in a synced folder."; exit 1; }
[ -f "$TARGET" ] || { echo "ERROR: $TARGET not found. Create it from .env.example first, then re-run."; exit 1; }

python3 - "$SOURCE" "$TARGET" <<'PY'
import re, sys
src, tgt = sys.argv[1], sys.argv[2]

def kv(path):
    d = {}
    for line in open(path):
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, _, v = s.partition("="); d[k.strip()] = v.strip()
    return d

bunny = {k: v for k, v in kv(src).items() if k.startswith("BUNNY")}
if not bunny:
    sys.exit(f"no BUNNY_* vars found in {src}")

txt = open(tgt).read()

def set_var(text, k, v):
    if re.search(rf'^{re.escape(k)}=', text, re.M):
        return re.sub(rf'^{re.escape(k)}=.*$', f'{k}={v}', text, flags=re.M), "updated"
    return text.rstrip("\n") + f"\n{k}={v}\n", "added"

changes = []
for k, v in bunny.items():
    txt, how = set_var(txt, k, v); changes.append((how, k))
# Local marker: never push to the dev/prod vectorstore prefix from a laptop.
txt, how = set_var(txt, "DEPLOY_ENV", "local"); changes.append((how, "DEPLOY_ENV=local"))

open(tgt, "w").write(txt)
for how, k in changes:
    print(f"  {how}: {k}")
print(f"done: {len(bunny)} Bunny var(s) from {src} -> {tgt}")
PY

echo ""
echo "Verify the app sees it:  docker compose up -d backend  (then scripts/test_storage_flow.sh)"
