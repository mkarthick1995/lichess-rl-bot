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
    """Return a summary of window.lichess.round.data — what the page knows."""
    try:
        snap = await page.evaluate(
            """() => {
                try {
                    var r = window.lichess && window.lichess.round;
                    if (!r || !r.data) return { ok: false, reason: 'no window.lichess.round.data' };
                    var d = r.data;
                    return {
                        ok: true,
                        game: d.game ? {
                            id: d.game.id,
                            status: d.game.status,
                            turns: d.game.turns,
                            initialFen: d.game.initialFen || null,
                            fen: d.game.fen || null,
                            speed: d.game.speed,
                            variant: d.game.variant && d.game.variant.key
                        } : null,
                        player: d.player ? { color: d.player.color, user: d.player.user && d.player.user.username } : null,
                        opponent: d.opponent ? { color: d.opponent.color, user: d.opponent.user && d.opponent.user.username, ai: d.opponent.ai } : null,
                        n_steps: d.steps ? d.steps.length : null,
                        last_uci: d.steps && d.steps.length > 1 ? d.steps[d.steps.length - 1].uci : null
                    };
                } catch(e) { return { ok: false, error: String(e) }; }
            }"""
        )
        log.debug("lichess state: " + json.dumps(snap, default=str))
        return snap
    except Exception as e:
        log.warning(f"lichess state dump failed: {e}")
        return None
