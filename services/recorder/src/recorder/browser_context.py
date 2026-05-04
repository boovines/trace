"""Active-tab URL + title capture for clicks inside a known browser.

Step 6.1 of the open follow-ups: the browser_dom tier in Step 4 needs
the synthesizer to know the URL the user was on when they clicked, so
it can emit a meaningful ``url_pattern`` on browser_dom execution
hints. The recorder is the only layer that observes the *moment* of
the click, so it owns this capture.

The implementation is **AppleScript via ``NSAppleScript``** rather
than AX for two reasons:

1. AX exposes ``AXURL`` only on the document/web-area element, which
   varies in availability across browsers and even across Chrome
   builds. AppleScript is uniformly supported by every scriptable
   browser and returns the canonical URL of the active tab.
2. AppleScript dispatch is per-bundle: we call the right script for
   the frontmost browser, so we never query a non-browser app and
   never accidentally drive the wrong app.

What this module does NOT do:

* **DOM-stable selectors** — Chrome exposes ``AXDOMIdentifier`` and
  ``AXDOMClassList`` on AX nodes; the resolver in
  :mod:`recorder.ax_resolver` will pick those up opportunistically in
  a follow-up. They're additive on the ``target`` dict and don't
  belong on the browser-context envelope.
* **History or full-tab metadata** — only the active tab's URL +
  title matter for synth purposes. Anything richer would slow the
  click hot path.

Permissions: AppleScript dispatch into another app requires Apple
Events permission (granted via macOS Privacy & Security settings the
first time the user records a browser session). When the permission
is missing, ``NSAppleScript`` returns an error dict — we treat that
the same as any other failure and return ``None`` so recording
continues.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import TypedDict

__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "KNOWN_BROWSER_BUNDLES",
    "BrowserContext",
    "is_browser_bundle",
    "resolve_browser_context",
]

logger = logging.getLogger(__name__)

#: Hard wall-clock timeout for one AppleScript dispatch. AppleScript
#: into a busy browser can take 50-150ms; we cap at 200ms so a hung
#: browser process never blocks the click hot path. On timeout the
#: capture is reported as ``None`` and recording continues.
DEFAULT_TIMEOUT_SECONDS: float = 0.2


class BrowserContext(TypedDict):
    """Trajectory-schema compatible browser-context envelope."""

    url: str
    title: str | None


#: AppleScript snippets that return a single string of the form
#: "<url>\n<title>" (or just "<url>" when the title can't be queried).
#: ``run`` joins lines with explicit newlines so the parsing layer
#: doesn't have to know AppleScript-side variable handling.
#:
#: Each bundle id maps to a script that targets that specific app —
#: dispatching to the wrong app is harmless (returns no match) but
#: dispatching to *any* non-frontmost app would surprise the user, so
#: the caller is responsible for verifying the frontmost-app id
#: matches before invoking ``run``.
_SCRIPTS_BY_BUNDLE: dict[str, str] = {
    # Chrome family (Chrome, Chrome Canary, Chrome Beta, Brave, Edge):
    # all expose the same scripting dictionary inherited from Chromium.
    "com.google.Chrome": (
        'tell application id "com.google.Chrome"\n'
        '  if (count of windows) = 0 then return ""\n'
        '  set t to active tab of front window\n'
        '  return (URL of t) & linefeed & (title of t)\n'
        "end tell"
    ),
    "com.google.Chrome.canary": (
        'tell application id "com.google.Chrome.canary"\n'
        '  if (count of windows) = 0 then return ""\n'
        '  set t to active tab of front window\n'
        '  return (URL of t) & linefeed & (title of t)\n'
        "end tell"
    ),
    "com.google.Chrome.beta": (
        'tell application id "com.google.Chrome.beta"\n'
        '  if (count of windows) = 0 then return ""\n'
        '  set t to active tab of front window\n'
        '  return (URL of t) & linefeed & (title of t)\n'
        "end tell"
    ),
    "com.brave.Browser": (
        'tell application id "com.brave.Browser"\n'
        '  if (count of windows) = 0 then return ""\n'
        '  set t to active tab of front window\n'
        '  return (URL of t) & linefeed & (title of t)\n'
        "end tell"
    ),
    "com.microsoft.edgemac": (
        'tell application id "com.microsoft.edgemac"\n'
        '  if (count of windows) = 0 then return ""\n'
        '  set t to active tab of front window\n'
        '  return (URL of t) & linefeed & (title of t)\n'
        "end tell"
    ),
    # Arc reuses the Chromium scripting model.
    "company.thebrowser.Browser": (
        'tell application id "company.thebrowser.Browser"\n'
        '  if (count of windows) = 0 then return ""\n'
        '  set t to active tab of front window\n'
        '  return (URL of t) & linefeed & (title of t)\n'
        "end tell"
    ),
    # Safari uses a different scripting dictionary (front document, not
    # active tab of front window).
    "com.apple.Safari": (
        'tell application id "com.apple.Safari"\n'
        '  if (count of documents) = 0 then return ""\n'
        '  set d to front document\n'
        '  return (URL of d) & linefeed & (name of d)\n'
        "end tell"
    ),
    # Firefox does NOT support AppleScript URL queries today (its
    # scripting dictionary doesn't expose tabs). We list the bundle
    # so ``is_browser_bundle`` returns True (recording still works);
    # ``resolve_browser_context`` returns None for it.
    "org.mozilla.firefox": "",
}

#: Frozenset of every bundle id this module recognises as a browser.
#: Exposed for ``recorder.session`` to short-circuit non-browser
#: events without paying the AppleScript dispatch cost.
KNOWN_BROWSER_BUNDLES: frozenset[str] = frozenset(_SCRIPTS_BY_BUNDLE)


def is_browser_bundle(bundle_id: str | None) -> bool:
    """Return True when ``bundle_id`` is in :data:`KNOWN_BROWSER_BUNDLES`."""
    return bundle_id is not None and bundle_id in KNOWN_BROWSER_BUNDLES


#: Type alias for a function that runs an AppleScript source string and
#: returns the result string (or raises). Tests inject a fake; real
#: callers get ``_run_applescript_via_nsapplescript`` below.
AppleScriptRunner = Callable[[str, float], str]


def resolve_browser_context(
    bundle_id: str | None,
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    runner: AppleScriptRunner | None = None,
) -> BrowserContext | None:
    """Return the active tab's URL+title for ``bundle_id``, or ``None``.

    ``None`` covers every "no useful capture" case so the click hot
    path treats them uniformly:

    * ``bundle_id`` is not a known browser.
    * The browser is one we know about but doesn't expose an
      AppleScript URL query (Firefox today).
    * The query timed out, raised, or returned an empty string (no
      open windows / Apple Events permission denied).
    * The result didn't parse into a non-empty URL.

    ``runner`` is for tests to inject a fake. Production calls fall
    through to :func:`_run_applescript_via_nsapplescript` which uses
    PyObjC's ``NSAppleScript``.
    """
    if not is_browser_bundle(bundle_id):
        return None
    assert bundle_id is not None  # narrowed by is_browser_bundle
    script = _SCRIPTS_BY_BUNDLE[bundle_id]
    if not script:  # Firefox-style "registered but unscriptable"
        return None

    invoker = runner or _run_applescript_via_nsapplescript
    try:
        result = invoker(script, timeout_seconds)
    except Exception as exc:
        logger.debug(
            "browser_context AppleScript failed for %s: %s", bundle_id, exc
        )
        return None

    return _parse_result(result)


def _parse_result(result: str) -> BrowserContext | None:
    """Split an AppleScript ``\"<url>\\n<title>\"`` result into a context.

    Returns ``None`` when the URL line is empty (no open window or
    redacted private tab). Title is optional; trailing whitespace is
    stripped on both fields.
    """
    if not result:
        return None
    lines = [ln.strip() for ln in result.split("\n", 1)]
    url = lines[0]
    if not url:
        return None
    title = lines[1] if len(lines) > 1 and lines[1] else None
    return {"url": url, "title": title}


def _run_applescript_via_nsapplescript(script: str, timeout_seconds: float) -> str:
    """Real AppleScript runner — used in production.

    Wraps :class:`Foundation.NSAppleScript` with a hard wall-clock
    timeout. NSAppleScript itself blocks until the target app
    responds; the timeout is enforced by running the dispatch on a
    helper thread and waiting on a :class:`threading.Event`. On
    timeout the helper thread is left to finish in the background
    (its result is discarded) — orphaning a thread is preferable to
    blocking the recorder's event loop.
    """
    # Imported lazily so importing this module never forces the PyObjC
    # framework load on tests / non-macOS environments.
    from Foundation import NSAppleScript

    out: dict[str, str | Exception] = {}
    done = threading.Event()

    def _invoke() -> None:
        try:
            apple = NSAppleScript.alloc().initWithSource_(script)
            descriptor, error = apple.executeAndReturnError_(None)
            if error is not None:
                out["error"] = RuntimeError(
                    f"NSAppleScript error: {dict(error)}"
                )
                return
            text = descriptor.stringValue() if descriptor is not None else None
            out["value"] = text or ""
        except Exception as exc:
            out["error"] = exc
        finally:
            done.set()

    threading.Thread(
        target=_invoke,
        name="recorder-browser-context",
        daemon=True,
    ).start()
    if not done.wait(timeout_seconds):
        raise TimeoutError(
            f"AppleScript dispatch exceeded {timeout_seconds:.2f}s"
        )
    if isinstance(out.get("error"), Exception):
        raise out["error"]  # type: ignore[misc]
    value = out.get("value")
    return value if isinstance(value, str) else ""
