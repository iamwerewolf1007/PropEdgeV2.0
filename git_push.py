"""
PropEdge V2.0 — git_push.py
────────────────────────────────────────────────────────────────────────────────
GitHub push via REST API (HTTPS + token). SSH is not used — SSH agent is not
forwarded in launchd environments and times out silently.

TOKEN SETUP (one-time, run in Terminal):
  pip3 install keyring
  python3 -c "import keyring; keyring.set_password('propedge', 'github_token', 'ghp_YOUR_TOKEN')"

TOKEN PRIORITY (highest → lowest):
  1. macOS Keychain via keyring  — survives reboots, launchd-safe
  2. GITHUB_TOKEN environment variable
  3. ROOT/.github_token file     — first line only, must start with ghp_
  4. config.py GITHUB_TOKEN constant

FILES PUSHED:
  predict run:  data/today.json
  grade run:    data/today.json + data/season_2024_25.json + data/season_2025_26.json
────────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

from config import (
    ROOT, GITHUB_OWNER, GITHUB_REPO, GITHUB_BRANCH,
    PUSH_FILES_PREDICT, PUSH_FILES_GRADE,
)


def _get_token() -> str | None:
    """Retrieve GitHub token from keychain → env → file → config."""
    # 1. macOS Keychain
    try:
        import keyring
        t = keyring.get_password("propedge", "github_token")
        if t and t.strip().startswith("ghp_"):
            return t.strip()
    except Exception:
        pass

    # 2. Environment variable
    t = os.environ.get("GITHUB_TOKEN", "").strip()
    if t.startswith("ghp_"):
        return t

    # 3. Token file (first line only — guards against paste accidents)
    token_file = ROOT / ".github_token"
    if token_file.exists():
        try:
            raw = token_file.read_text().strip()
            first = raw.splitlines()[0].strip() if raw else ""
            if first.startswith("ghp_"):
                return first
            elif first:
                print(f"  ⚠ Git: .github_token found but wrong format: {first[:20]}...")
        except Exception:
            pass

    # 4. config.py constant
    try:
        from config import GITHUB_TOKEN  # type: ignore
        if GITHUB_TOKEN and GITHUB_TOKEN.startswith("ghp_"):
            return GITHUB_TOKEN.strip()
    except ImportError:
        pass

    return None


def _ssl_context():
    """Build SSL context — uses certifi if available (macOS python.org builds need this)."""
    try:
        import certifi, ssl
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        import ssl
        return ssl.create_default_context()


def _push_file(path: Path, token: str, message: str) -> bool:
    """Push a single file to GitHub via REST API. Returns True on success."""
    import urllib.request
    import urllib.error

    if not path.exists():
        return True

    size_mb = path.stat().st_size / 1024 / 1024
    if size_mb > 95:
        print(f"  ⚠ Git: {path.name} is {size_mb:.0f}MB — exceeds 100MB limit. Skipped.")
        return False
    if size_mb > 50:
        print(f"  ⚠ Git: {path.name} is {size_mb:.0f}MB — large file, may be slow...")

    try:
        b64 = base64.b64encode(path.read_bytes()).decode()
    except Exception as e:
        print(f"  ⚠ Git: read error {path.name}: {e}")
        return False

    rel     = path.relative_to(ROOT).as_posix()
    api_url = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{rel}"
    headers = {
        "Authorization": f"token {token}",
        "Content-Type":  "application/json",
        "Accept":        "application/vnd.github.v3+json",
        "User-Agent":    "PropEdge-V2",
    }
    ctx = _ssl_context()

    # Fetch current SHA (required to update existing file)
    sha = None
    try:
        req = urllib.request.Request(f"{api_url}?ref={GITHUB_BRANCH}", headers=headers)
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            sha = json.loads(resp.read()).get("sha")
    except urllib.error.HTTPError as e:
        if e.code != 404:
            print(f"  ⚠ Git: SHA fetch failed {path.name}: HTTP {e.code}")
            return False
    except Exception as e:
        print(f"  ⚠ Git: SHA fetch error {path.name}: {e}")
        return False

    payload: dict = {"message": message, "content": b64, "branch": GITHUB_BRANCH}
    if sha:
        payload["sha"] = sha

    try:
        req = urllib.request.Request(
            api_url, data=json.dumps(payload).encode(),
            headers=headers, method="PUT",
        )
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            resp.read()
        return True
    except urllib.error.HTTPError as e:
        print(f"  ⚠ Git: push failed {path.name}: {e.code} {e.read().decode()[:120]}")
        return False
    except Exception as e:
        print(f"  ⚠ Git: push error {path.name}: {e}")
        return False


def push(message: str, grade: bool = False, files: list[str] | None = None) -> None:
    """
    Push files to GitHub.

    Args:
        message: Commit message.
        grade:   True → push predict + season JSON (post-grade).
                 False → push predict files only.
        files:   Override file list with explicit relative paths.
    """
    token = _get_token()
    if not token:
        print("  ⚠ Git: No GitHub token found.")
        print("    Setup: python3 -c \"import keyring; keyring.set_password('propedge', 'github_token', 'ghp_YOUR_TOKEN')\"")
        return

    default = PUSH_FILES_GRADE if grade else PUSH_FILES_PREDICT
    targets = [ROOT / f for f in (files or default)]

    ok = fail = 0
    fail_names: list[str] = []

    for fpath in targets:
        if not fpath.exists():
            continue
        success = _push_file(fpath, token, message)
        if not success:
            time.sleep(3)
            success = _push_file(fpath, token, message)  # one retry
        if success:
            ok += 1
        else:
            fail += 1
            fail_names.append(fpath.name)

    existing = sum(1 for t in targets if t.exists())
    if fail_names:
        print(f"  ⚠ Git: {ok}/{existing} pushed — {fail} failed: {fail_names}")
    else:
        print(f"  ✓ Git: {ok}/{existing} files pushed — {message}")


def token_check() -> None:
    """
    Diagnostic: verify token and test against GitHub API.
    Run: python3 run.py token-check
    """
    import urllib.request, urllib.error

    print("\n  PropEdge V2.0 — GitHub Token Diagnostic")
    print("  " + "─" * 52)

    sources = []

    try:
        import keyring
        t = keyring.get_password("propedge", "github_token")
        if t and t.strip().startswith("ghp_"):
            sources.append(("Keychain", t.strip()[:12] + "..."))
        elif t:
            sources.append(("Keychain", f"BAD FORMAT: {t.strip()[:20]}"))
        else:
            sources.append(("Keychain", "NOT SET"))
    except ImportError:
        sources.append(("Keychain", "keyring not installed — pip3 install keyring"))

    env_t = os.environ.get("GITHUB_TOKEN", "").strip()
    sources.append(("Env GITHUB_TOKEN",
                    (env_t[:12] + "...") if env_t.startswith("ghp_") else
                    ("NOT SET" if not env_t else f"BAD FORMAT: {env_t[:20]}")))

    tf = ROOT / ".github_token"
    if tf.exists():
        raw = tf.read_text().strip()
        first = raw.splitlines()[0].strip() if raw else ""
        sources.append((".github_token",
                        (first[:12] + "...") if first.startswith("ghp_") else
                        f"BAD FORMAT: {first[:30]}"))
    else:
        sources.append((".github_token", "NOT FOUND"))

    for source, value in sources:
        sym = "✓" if "..." in value else "✗"
        print(f"  {sym} {source:<22} {value}")

    token = _get_token()
    print()
    if not token:
        print("  ✗ No valid token. Run:")
        print("    python3 -c \"import keyring; keyring.set_password('propedge', 'github_token', 'ghp_YOUR_TOKEN')\"")
        return

    print(f"  Active: {token[:12]}...{token[-4:]}")
    print(f"  Testing against GitHub API...")

    ctx = _ssl_context()
    api  = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}"
    hdrs = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "PropEdge-V2",
    }
    try:
        import urllib.request
        req = urllib.request.Request(api, headers=hdrs)
        with urllib.request.urlopen(req, timeout=10, context=ctx) as resp:
            data = json.loads(resp.read())
            print(f"  ✓ Token VALID — repo: {data.get('full_name')} | private: {data.get('private')}")
            print(f"  ✓ Run: python3 run.py predict")
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print(f"  ✗ Token REJECTED (401) — expired or revoked. Generate a new token.")
        elif e.code == 404:
            print(f"  ✗ Repo not found (404) — check GITHUB_OWNER/GITHUB_REPO in config.py")
        else:
            print(f"  ✗ HTTP {e.code}: {e.read().decode()[:200]}")
    except Exception as e:
        print(f"  ✗ Connection error: {e}")
