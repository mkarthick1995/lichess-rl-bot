"""Minimal client for the official Lichess Bot API (BOT accounts only).

Docs: https://lichess.org/api#tag/Bot
The token needs the `bot:play` scope. It is read from the LICHESS_TOKEN
environment variable or from a token file (default `.lichess-token`, which
is git-ignored) and is never logged.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Iterator, Optional

import requests

log = logging.getLogger("lichess.api")

BASE = "https://lichess.org"


def load_token(path: str | Path = ".lichess-token") -> str:
    tok = os.environ.get("LICHESS_TOKEN", "").strip()
    if not tok and Path(path).exists():
        tok = Path(path).read_text(encoding="utf-8").strip()
    if not tok:
        raise SystemExit(
            "No Lichess API token. Create one at https://lichess.org/account/oauth/token "
            "with the 'bot:play' scope (logged in as the BOT account), then either set "
            "LICHESS_TOKEN or save it in .lichess-token in the project folder.")
    return tok


class LichessAPI:
    def __init__(self, token: str, base: str = BASE, timeout: float = 15.0):
        self.base = base
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {token}",
                               "User-Agent": "lichess-rl-bot (bot api client)"})

    # -- plumbing -------------------------------------------------------------
    def _req(self, method: str, path: str, retries: int = 3, **kw) -> requests.Response:
        for attempt in range(retries):
            try:
                r = self.s.request(method, self.base + path, timeout=self.timeout, **kw)
            except requests.RequestException as e:
                log.warning(f"{method} {path} failed ({e}); retry {attempt + 1}")
                time.sleep(1 + attempt)
                continue
            if r.status_code == 429:
                # Lichess asks clients to wait a full minute after a 429.
                log.warning(f"{method} {path}: rate limited (429); waiting 60 s")
                time.sleep(60)
                continue
            return r
        raise RuntimeError(f"{method} {path}: giving up after {retries} attempts")

    def _post(self, path: str, data: Optional[dict] = None) -> bool:
        r = self._req("POST", path, data=data or {})
        if r.status_code != 200:
            log.warning(f"POST {path} → {r.status_code}: {r.text[:200]}")
        return r.status_code == 200

    def _stream(self, path: str) -> Iterator[dict]:
        """ndjson stream; blank keep-alive lines are skipped. Returns when the
        server closes the stream."""
        with self.s.get(self.base + path, stream=True, timeout=(self.timeout, 120)) as r:
            if r.status_code == 429:
                log.warning(f"GET {path}: rate limited (429); waiting 60 s")
                time.sleep(60)
                return
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    yield json.loads(line)

    # -- account --------------------------------------------------------------
    def account(self) -> dict:
        r = self._req("GET", "/api/account")
        r.raise_for_status()
        return r.json()

    def upgrade_to_bot(self) -> bool:
        """IRREVERSIBLE: turns the token's account into a BOT account. Only
        works for an account that has never played a game."""
        return self._post("/api/bot/account/upgrade")

    # -- streams --------------------------------------------------------------
    def stream_events(self) -> Iterator[dict]:
        return self._stream("/api/stream/event")

    def stream_game(self, game_id: str) -> Iterator[dict]:
        return self._stream(f"/api/bot/game/stream/{game_id}")

    def online_bots(self, nb: int = 50) -> list[dict]:
        r = self._req("GET", f"/api/bot/online?nb={nb}")
        if r.status_code != 200:
            return []
        return [json.loads(l) for l in r.text.splitlines() if l.strip()]

    # -- game actions ---------------------------------------------------------
    def make_move(self, game_id: str, uci: str) -> bool:
        return self._post(f"/api/bot/game/{game_id}/move/{uci}")

    def abort(self, game_id: str) -> bool:
        return self._post(f"/api/bot/game/{game_id}/abort")

    def chat(self, game_id: str, text: str, room: str = "player") -> bool:
        return self._post(f"/api/bot/game/{game_id}/chat", {"room": room, "text": text})

    # -- challenges -----------------------------------------------------------
    def accept(self, challenge_id: str) -> bool:
        return self._post(f"/api/challenge/{challenge_id}/accept")

    def decline(self, challenge_id: str, reason: str = "generic") -> bool:
        return self._post(f"/api/challenge/{challenge_id}/decline", {"reason": reason})

    def challenge(self, username: str, limit_s: int, inc_s: int, rated: bool,
                  color: str = "random") -> Optional[str]:
        r = self._req("POST", f"/api/challenge/{username}",
                      data={"rated": str(rated).lower(), "clock.limit": limit_s,
                            "clock.increment": inc_s, "color": color, "variant": "standard"})
        if r.status_code != 200:
            log.warning(f"challenge {username} → {r.status_code}: {r.text[:200]}")
            return None
        return r.json().get("challenge", r.json()).get("id")

    def cancel_challenge(self, challenge_id: str) -> bool:
        return self._post(f"/api/challenge/{challenge_id}/cancel")
