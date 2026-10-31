"""Debug helpers for the Lichess wiring: dump page state, save screenshots.

These exist so that when something misfires, we can save artefacts to disk and
the user can paste log lines (or attach the screenshot) for diagnosis.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("lichess.debug")

_SCREENSHOT_DIR = Path("logs/screenshots")


async def save_screenshot(page, tag: str = "error") -> Path | None:
    """Save a full-page screenshot. Returns the path, or None on failure."""
    try:
        _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
        path = _SCREENSHOT_DIR / f"{tag}-{ts}.png"
        await page.screenshot(path=str(path), full_page=True)
        log.info(f"screenshot saved: {path}")
        return path
    except Exception as e:
        log.warning(f"screenshot failed: {e}")
        return None


async def dump_lichess_state(page) -> dict | None:
    """Summarise the page-init-data JSON + body playing-state — what the page
    knows at load. Useful when a parse looks wrong."""
    try:
        snap = await page.evaluate(
            """() => {
                try {
                    var out = { playing: document.body.classList.contains('playing') };
                    var el = document.getElementById('page-init-data');
                    if (!el) { out.ok = false; out.reason = 'no #page-init-data'; return out; }
                    var d = JSON.parse(el.textContent).data;
                    out.ok = true;
                    out.game = d.game ? {
                        id: d.game.id, status: d.game.status, turns: d.game.turns,
                        fen: d.game.fen || null, speed: d.game.speed,
                        variant: d.game.variant && d.game.variant.key
                    } : null;
                    out.player = d.player ? { color: d.player.color } : null;
                    out.opponent = d.opponent ? { color: d.opponent.color, ai: d.opponent.ai } : null;
                    out.n_steps = d.steps ? d.steps.length : null;
                    return out;
                } catch(e) { return { ok: false, error: String(e) }; }
            }"""
        )
        log.debug("lichess state: " + json.dumps(snap, default=str))
        return snap
    except Exception as e:
        log.warning(f"lichess state dump failed: {e}")
        return None
