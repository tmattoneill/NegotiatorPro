#!/usr/bin/env bash
#
# test_storage_flow.sh — local L2 check for the Bunny storage write-through.
#
# Exercises the real admin path end to end:
#   admin login -> upload a throwaway source -> verify it landed in the Bunny
#   corpus zone -> delete via the API -> verify it's gone.
#
# Run from anywhere AFTER the app is up (local Docker stack or uvicorn):
#   scripts/test_storage_flow.sh
#   BASE_URL=http://localhost:8000 scripts/test_storage_flow.sh     # default
#   BASE_URL=https://dev.amfonica.com scripts/test_storage_flow.sh  # or dev
#
# Reads ADMIN_PASSWORD + BUNNY_* from .env (override with ENV_FILE=...). The
# Bunny checks use backend/storage.py, which is stdlib-only, so any python3
# works. Uses a uniquely-named throwaway file and always cleans it up — but note
# the CORPUS zone is shared, so don't point this at a zone you don't want touched.

set -euo pipefail
cd "$(dirname "$0")/.."

BASE_URL="${BASE_URL:-http://localhost:8000}"
ENV_FILE="${ENV_FILE:-.env}"
TESTNAME="_storage_selftest_$(date +%s).txt"

[ -f "$ENV_FILE" ] || { echo "ERROR: $ENV_FILE not found"; exit 1; }

# Load env safely (shlex-quoted) so curl sees ADMIN_PASSWORD and python sees BUNNY_*.
eval "$(python3 - "$ENV_FILE" <<'PY'
import shlex, sys
for line in open(sys.argv[1]):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        print(f"export {k.strip()}={shlex.quote(v.strip())}")
PY
)"

pass() { printf '  \033[1;32mPASS\033[0m %s\n' "$1"; }
fail() { printf '  \033[1;31mFAIL\033[0m %s\n' "$1"; cleanup; exit 1; }
cleanup() { rm -f "/tmp/$TESTNAME"; }

# Query the corpus zone via storage.py loaded DIRECTLY (avoids backend/__init__.py
# which imports langchain etc.). storage.py is stdlib-only. Reads BUNNY_* from env.
#   bunny configured | bunny exists <name> | bunny absent <name>
bunny() {
  python3 - "$@" <<'PY'
import importlib.util, sys
spec = importlib.util.spec_from_file_location("storage", "backend/storage.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
corpus = m.corpus
op = sys.argv[1]
if op == "configured":
    sys.exit(0 if corpus.is_configured() and corpus.can_write() else 1)
if op == "exists":
    sys.exit(0 if corpus.exists(sys.argv[2]) else 1)
if op == "absent":
    sys.exit(0 if not corpus.exists(sys.argv[2]) else 1)
sys.exit(2)
PY
}

echo "== storage flow test -> $BASE_URL  (DEPLOY_ENV=${DEPLOY_ENV:-unset}) =="

# 0) The test is only meaningful if the corpus store is configured + writable.
bunny configured \
  && pass "corpus store configured + writable" \
  || fail "corpus store not configured/writable — set BUNNY_* in $ENV_FILE"

# 1) admin login
TOKEN=$(curl -s -X POST "$BASE_URL/api/auth/login" -H "Content-Type: application/json" \
  -d "{\"username\":\"admin\",\"password\":\"${ADMIN_PASSWORD:-}\"}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))" 2>/dev/null || true)
[ -n "$TOKEN" ] && pass "admin login" || fail "admin login (is the app up? is ADMIN_PASSWORD correct?)"

# 2) upload a throwaway source via the admin API
printf 'storage flow self-test %s\n' "$(date -u)" > "/tmp/$TESTNAME"
UP=$(curl -s -X POST "$BASE_URL/api/admin/sources/upload" -H "Authorization: Bearer $TOKEN" -F "file=@/tmp/$TESTNAME")
echo "$UP" | python3 -c "import sys,json; sys.exit(0 if json.load(sys.stdin).get('success') else 1)" \
  && pass "upload via admin API" || fail "upload failed: $UP"

# 3) verify it landed in the Bunny corpus zone
bunny exists "$TESTNAME" \
  && pass "present in Bunny corpus zone" || fail "not found in Bunny after upload"

# 4) delete via the admin API
DEL=$(curl -s -X DELETE "$BASE_URL/api/admin/sources/$TESTNAME" -H "Authorization: Bearer $TOKEN")
echo "$DEL" | python3 -c "import sys,json; sys.exit(0 if json.load(sys.stdin).get('remote_deleted') else 1)" \
  && pass "delete via admin API (remote_deleted=true)" || fail "delete failed: $DEL"

# 5) verify it's gone from Bunny
bunny absent "$TESTNAME" \
  && pass "gone from Bunny corpus zone" || fail "still present in Bunny after delete"

cleanup
printf '\033[1;32mALL CHECKS PASSED\033[0m\n'
