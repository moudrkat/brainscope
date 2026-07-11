"""Browser tests (Playwright + chromium, headless): the viz page against a
real server running the tiny random model — tabs, live generation, J-lens
grid, trace replay, steering-from-word. Marked `ui`; run with
`pytest -m ui` after `playwright install chromium`."""

import socket
import threading
import time

import pytest
import uvicorn

from brainscope import server as bs
from tests.conftest import make_state

pytest.importorskip("playwright.sync_api")
from playwright.sync_api import expect, sync_playwright  # noqa: E402

pytestmark = pytest.mark.ui


@pytest.fixture(scope="module")
def base_url(model, tok, fitted_lens, tmp_path_factory):
    make_state(model, tok, tmp_path_factory.mktemp("traces"))
    bs.state["jlens"], bs.state["jlens_on"] = fitted_lens, True
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    cfg = uvicorn.Config(bs.app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 15
    import urllib.request
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/info", timeout=1)
            break
        except OSError:
            time.sleep(0.2)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True


@pytest.fixture(scope="module")
def page(base_url):
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(base_url)
        yield page
        browser.close()


def test_page_boots_with_new_controls(page):
    expect(page.locator("#modelinfo")).to_contain_text("tiny-random-qwen2")
    expect(page.locator("#tab-jlens")).to_be_visible()
    expect(page.locator("#tab-traces")).to_be_visible()
    expect(page.locator("#jlensbtn")).to_contain_text("j-lens on")
    expect(page.locator("#jwordbox")).to_be_visible()


def test_generation_fills_text_and_both_lens_grids(page):
    page.fill("#msg", "Hello little model")
    page.click("#send")
    expect(page.locator("#text .tok").first).to_be_visible(timeout=60_000)
    expect(page.locator("#stxt")).to_contain_text("done", timeout=60_000)
    page.click('.tab[data-p="lens"]')
    expect(page.locator("#lensgrid .lenscol:not(.axis)").first).to_be_visible()
    page.click("#tab-jlens")
    expect(page.locator("#jlensgrid .lenscol:not(.axis)").first).to_be_visible()
    # lens cells carry the rich tooltip payload
    n = page.locator("#jlensgrid .lenscol:not(.axis)").count()
    assert n >= 1


def test_traces_tab_lists_and_replays(page):
    page.click("#tab-traces")
    row = page.locator(".trrow").first
    expect(row).to_be_visible(timeout=10_000)
    row.click()
    expect(page.locator("#trdetail")).to_be_visible()
    expect(page.locator("#trtext .tok").first).to_be_visible()
    expect(page.locator("#trstep")).to_contain_text("token")
    # scrub to the start: current-token marker moves; token 0 is prefill,
    # so no lens column there — that's by design
    page.locator("#trscrub").fill("0")
    page.locator("#trscrub").dispatch_event("input")
    expect(page.locator("#trtext .tok.cur")).to_have_count(1)
    expect(page.locator("#trstep")).to_contain_text("prefill")
    # a captured step shows the per-step lens column(s)
    page.locator("#trscrub").fill("3")
    page.locator("#trscrub").dispatch_event("input")
    assert page.locator("#trlenses .lensgrid").count() >= 1
    # emergence canvas got drawn (tracking line present in JS state)
    assert page.evaluate("trEmergence !== null")


def test_hidden_toggle_roundtrips(page):
    page.click("#tab-traces")
    btn = page.locator("#trhidden")
    before = btn.text_content()
    btn.click()
    expect(btn).not_to_have_text(before)
    btn.click()
    expect(btn).to_have_text(before)


def test_steer_from_word_creates_and_selects_direction(page):
    page.fill("#jword", "cake")
    page.click("#jwordadd")
    expect(page.locator("#steerbox")).to_be_visible(timeout=10_000)
    expect(page.locator("#dir")).to_have_value("j:cake")
    # preset from the server prefilled strength + layer range
    assert page.locator("#lrange").input_value() != ""
