"""Automate starting a new "Play with the computer" (Stockfish) game on
Lichess, so the farm loop (src/lichess/farm.py) doesn't need a human to
click through the setup screen for every game.

USE AGAINST COMPUTER OPPONENTS ONLY. This module only ever starts games
against Lichess's built-in Stockfish AI, never against human players.

Lichess's AI setup is the "Game setup" <dialog> opened by
https://lichess.org/#ai (no server round-trip until "Play against computer" is
clicked). As of 2026-10 it uses radio inputs #sf_level_1..8 and
#color-picker-{white,random,black} with <label for=…>, and the button
.lobby__start__button--ai; clicking it navigates to the new game's URL. Because this page isn't
designed for scripted automation and its DOM has changed before (see
reader.py's docstring about the page-init-data migration), this module
follows the same defensive, multi-strategy pattern used in reader.py and
executor.py: try a specific selector, log and fall back if it's missing, and
save a screenshot on total failure so the selectors can be fixed from the
artifact rather than blind guessing.

If Lichess's markup drifts, the first thing to check is `_debug.save_screenshot`
output for the tag "lobby-*" — the DOM structure of the setup screen is the
single most likely thing to need updating here.
"""
from __future__ import annotations

import asyncio
import logging
import re

from playwright.async_api import Page

from ._debug import save_screenshot

log = logging.getLogger("lichess.lobby")

_SETUP_URL = "https://lichess.org/#ai"


async def _select_level(page: Page, level: int) -> bool:
    level = max(1, min(8, int(level)))

    # Strategy 0 — current setup dialog: radio #sf_level_N with a <label for>.
    try:
        lbl = page.locator(f"label[for='sf_level_{level}']")
        if await lbl.count() > 0:
            await lbl.first.click(timeout=5000)
            if await page.locator(f"#sf_level_{level}").is_checked():
                log.info(f"lobby: selected level {level} via #sf_level_{level}")
                return True
    except Exception as e:
        log.debug(f"lobby: #sf_level radio strategy raised: {e}")

    # Strategy 1 — legacy <select id="config_level"> with numeric <option>s.
    try:
        sel = page.locator("#config_level")
        if await sel.count() > 0:
            await sel.select_option(str(level))
            log.info(f"lobby: selected level {level} via #config_level <select>")
            return True
    except Exception as e:
        log.debug(f"lobby: #config_level select strategy raised: {e}")

    # Strategy 2 — a group of clickable level buttons/radios with the level
    # number as their visible text (current Lichess setup-screen style).
    try:
        candidates = page.locator(
            "group.radio label, .level-group label, .buttons label, label"
        )
        n = await candidates.count()
        for i in range(n):
            el = candidates.nth(i)
            text = (await el.inner_text()).strip()
            if text == str(level):
                await el.click()
                log.info(f"lobby: selected level {level} via label click (candidate {i})")
                return True
    except Exception as e:
        log.debug(f"lobby: label-click strategy raised: {e}")

    # Strategy 3 — JS scan: find any element whose sole trimmed text content
    # is exactly the level number, inside a container that looks like a
    # difficulty/level picker.
    try:
        clicked = await page.evaluate(
            f"""() => {{
                var containers = document.querySelectorAll(
                    'group, .level, .buttons, form'
                );
                for (var c = 0; c < containers.length; c++) {{
                    var els = containers[c].querySelectorAll('label, button, a, span');
                    for (var i = 0; i < els.length; i++) {{
                        if ((els[i].textContent || '').trim() === '{level}') {{
                            els[i].click();
                            return true;
                        }}
                    }}
                }}
                return false;
            }}"""
        )
        if clicked:
            log.info(f"lobby: selected level {level} via JS text-match scan")
            return True
    except Exception as e:
        log.debug(f"lobby: JS text-match strategy raised: {e}")

    log.warning(f"lobby: could not find a level-{level} selector; "
                f"leaving level at whatever the page defaults to")
    return False


async def _select_color(page: Page, color: str) -> bool:
    """color: 'white', 'black', or 'random'."""
    color = color.lower()
    if color not in ("white", "black", "random"):
        color = "random"

    # Strategy 1 — current dialog's label[for=color-picker-<name>], then the
    # legacy id="color-<name>" (white/black/random) buttons.
    for sel in (f"label[for='color-picker-{color}']", f"#color-{color}", f"[data-color='{color}']", f".color-{color}"):
        try:
            el = page.locator(sel)
            if await el.count() > 0:
                await el.first.click(timeout=5000)
                log.info(f"lobby: selected color={color} via {sel}")
                return True
        except Exception as e:
            log.debug(f"lobby: color selector {sel} raised: {e}")

    # Strategy 2 — JS scan by title/aria-label/text containing the colour name.
    try:
        clicked = await page.evaluate(
            f"""() => {{
                var els = document.querySelectorAll('button, a, label, span');
                for (var i = 0; i < els.length; i++) {{
                    var t = ((els[i].getAttribute('title') || '') + ' ' +
                             (els[i].getAttribute('aria-label') || '') + ' ' +
                             (els[i].textContent || '')).toLowerCase();
                    if (t.indexOf('{color}') !== -1) {{
                        els[i].click();
                        return true;
                    }}
                }}
                return false;
            }}"""
        )
        if clicked:
            log.info(f"lobby: selected color={color} via JS text scan")
            return True
    except Exception as e:
        log.debug(f"lobby: color JS-scan strategy raised: {e}")

    log.warning(f"lobby: could not find a color={color} selector; using page default")
    return False


async def _click_play(page: Page) -> bool:
    # Scope to the dialog: the home page has its own (hidden behind the
    # modal) "play with the computer" button with the same class.
    for sel in ("dialog .lobby__start__button--ai", "dialog button:has-text('Play')",
                "button:has-text('Play')", "form button[type='submit']", ".submit button"):
        try:
            el = page.locator(sel)
            if await el.count() > 0:
                await el.first.click(timeout=5000)
                log.info(f"lobby: clicked Play via {sel}")
                return True
        except Exception as e:
            log.debug(f"lobby: play-button selector {sel} raised: {e}")
    log.error("lobby: could not find a 'Play' button")
    return False


async def start_stockfish_game(page: Page, *, level: int, color: str = "random",
                               timeout: float = 30.0) -> bool:
    """Navigate to the AI setup screen, pick `level` (1-8) and `color`, click
    Play, and wait for the board to appear. Returns True on success.

    Best-effort: Lichess's setup-screen DOM isn't a stable API and this is
    expected to need selector maintenance over time (see module docstring).
    """
    log.info(f"lobby: starting new game vs Stockfish level={level} color={color}")
    try:
        await page.goto(_SETUP_URL, wait_until="domcontentloaded")
        await asyncio.sleep(1.0)  # let the setup screen's JS finish rendering
    except Exception as e:
        log.error(f"lobby: navigation to {_SETUP_URL} failed: {e}")
        await save_screenshot(page, tag="lobby-nav-fail")
        return False

    await _select_level(page, level)
    await asyncio.sleep(0.2)
    await _select_color(page, color)
    await asyncio.sleep(0.2)

    if not await _click_play(page):
        await save_screenshot(page, tag="lobby-no-play-button")
        return False

    try:
        # The home page has mini-boards of its own, so wait for the game URL
        # (lichess.org/<8-12 char id>) before waiting for the board.
        await page.wait_for_url(re.compile(r"lichess\.org/[A-Za-z0-9]{8,12}(/(white|black))?$"),
                                timeout=int(timeout * 1000))
        await page.wait_for_selector(".round__app cg-board, main.round cg-board, cg-board",
                                     timeout=int(timeout * 1000))
        log.info(f"lobby: game started at {page.url}")
        return True
    except Exception:
        log.error(f"lobby: cg-board did not appear within {timeout:.0f}s after clicking Play")
        await save_screenshot(page, tag="lobby-no-board")
        return False
