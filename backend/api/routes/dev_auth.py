"""
Dev-site pre-authentication.

Provides a cookie-based gate for dev.amfonica.com so the site is accessible
from anywhere without prompting for credentials on every page load.

Flow:
  1. nginx auth_request → GET /api/dev-auth/check
     - returns 200 if the cookie is valid, 401 otherwise
  2. nginx error_page 401 → redirect to /dev-auth/login
  3. User submits the password form → POST /api/dev-auth/login
     - on success: set signed cookie, redirect to original URL
  4. All subsequent page loads: cookie is present → check returns 200 → no prompt

The signed cookie is HMAC-SHA256(secret, unix_timestamp). Expiry is enforced
in the check handler, not by relying solely on the cookie max-age.

Only register this router in environments where DEV_AUTH_PASSWORD is set
(see main.py). In production (no nginx auth_request wired up) these routes
are harmless but unused.
"""

import hashlib
import hmac
import html as _html
import os
import time

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

router = APIRouter(prefix="/api/dev-auth", tags=["dev-auth"])

_PASSWORD = os.getenv("DEV_AUTH_PASSWORD", "amfonica123")
# Fall back to password itself so the system works without explicit secret,
# but you should set DEV_AUTH_SECRET to a random string in production.
_SECRET = os.getenv("DEV_AUTH_SECRET", _PASSWORD)
_COOKIE = "amfonica_dev"
_MAX_AGE = 30 * 24 * 3600  # 30 days

_LOGIN_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Amfonica Dev</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100dvh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: #0a0a0a;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #e5e5e5;
    }}
    .card {{
      background: #141414;
      border: 1px solid #2a2a2a;
      border-radius: 10px;
      padding: 2rem;
      width: 100%;
      max-width: 340px;
    }}
    h1 {{
      margin: 0 0 1.5rem;
      font-size: 1.1rem;
      font-weight: 600;
      letter-spacing: -0.01em;
      color: #fff;
    }}
    label {{
      display: block;
      font-size: 0.8rem;
      color: #888;
      margin-bottom: 0.35rem;
    }}
    input[type=password] {{
      width: 100%;
      padding: 0.55rem 0.75rem;
      background: #1e1e1e;
      border: 1px solid #333;
      border-radius: 6px;
      color: #e5e5e5;
      font-size: 0.95rem;
      outline: none;
      transition: border-color 0.15s;
    }}
    input[type=password]:focus {{ border-color: #A8C5FF; }}
    .error {{
      font-size: 0.8rem;
      color: #f87171;
      margin: 0 0 0.75rem;
    }}
    button {{
      margin-top: 1rem;
      width: 100%;
      padding: 0.6rem;
      background: #A8C5FF;
      color: #000;
      border: none;
      border-radius: 6px;
      font-size: 0.9rem;
      font-weight: 600;
      cursor: pointer;
    }}
    button:hover {{ background: #bdd4ff; }}
  </style>
</head>
<body>
  <div class="card">
    <h1>Amfonica Dev</h1>
    {error}
    <form method="post" action="/dev-auth/login">
      <input type="hidden" name="next" value="{next}" />
      <label for="pw">Password</label>
      <input id="pw" type="password" name="password" autofocus autocomplete="current-password" />
      <button type="submit">Sign in</button>
    </form>
  </div>
</body>
</html>
"""


def _sign(ts: str) -> str:
    return hmac.new(_SECRET.encode(), ts.encode(), hashlib.sha256).hexdigest()


def _make_token() -> str:
    ts = str(int(time.time()))
    return f"{ts}.{_sign(ts)}"


def _verify_token(token: str) -> bool:
    try:
        ts, sig = token.rsplit(".", 1)
        if not hmac.compare_digest(sig, _sign(ts)):
            return False
        if int(time.time()) - int(ts) > _MAX_AGE:
            return False
        return True
    except Exception:
        return False


@router.get("/login", response_class=HTMLResponse)
async def login_page(next: str = "/"):
    return _LOGIN_HTML.format(error="", next=_html.escape(next))


@router.post("/login")
async def login(password: str = Form(...), next: str = Form("/")):
    if password != _PASSWORD:
        return HTMLResponse(
            _LOGIN_HTML.format(
                error='<p class="error">Wrong password.</p>',
                next=_html.escape(next),
            ),
            status_code=401,
        )
    safe_next = next if next.startswith("/") and not next.startswith("//") else "/"
    response = RedirectResponse(url=safe_next, status_code=303)
    response.set_cookie(
        _COOKIE,
        _make_token(),
        max_age=_MAX_AGE,
        httponly=True,
        samesite="lax",
        secure=True,
    )
    return response


@router.get("/check")
async def check(request: Request):
    token = request.cookies.get(_COOKIE)
    if token and _verify_token(token):
        return Response(status_code=200)
    return Response(status_code=401)
