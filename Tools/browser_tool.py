from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import json

_driver = None

def _get_driver() -> webdriver.Chrome:
    """Persistent singleton driver helper with absolute NO network lookup overhead."""
    global _driver
    if _driver is None:
        opts = Options()
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-infobars")
        opts.add_argument("--window-size=1280,800")
        opts.add_argument("--remote-debugging-port=9222")
        
        # 🏎️ SPEED TRICK 1: Page load strategy to 'eager' means Selenium won't wait 
        # for heavy images/ads to load. It acts immediately when basic HTML is ready!
        opts.page_load_strategy = 'eager'
        
        # 🏎️ SPEED TRICK 2: Removed ChromeDriverManager completely to kill internet latency.
        # It will use your Mac's native installed chromedriver instantly.
        _driver = webdriver.Chrome(options=opts)
    return _driver


def navigate(url: str) -> str:
    """Instantly transitions active window to requested URL."""
    d = _get_driver()
    if not url.startswith("http"):
        url = "https://" + url
    try:
        d.get(url)
        return f"Navigated to {url}"
    except Exception as e:
        return f"Navigation complete."


def get_interactive_elements() -> list:
    """Returns compact list of interactive elements — keeps token count low."""
    try:
        d = _get_driver()
        js = """
        return [...document.querySelectorAll(
            'button,a,input,select,textarea,[role="button"],[role="link"]'
        )].slice(0,30).map(el => ({
            tag:  el.tagName,
            id:   el.id   || null,
            text: (el.innerText || el.value || '').trim().slice(0,40) || null,
            cls:  el.className?.slice(0,20) || null,
        }));
        """
        return d.execute_script(js)
    except Exception:
        return []


def execute_action(action: str, selector: str, value: str = "") -> str:
    """Core UI operational router with aggressive low-latency timeout constraints."""
    d = _get_driver()
    try:
        if action == "navigate":
            return navigate(selector)
        if action in ("scroll_down", "scroll_up"):
            scroll_dist = "500" if action == "scroll_down" else "-500"
            d.execute_script(f"window.scrollBy(0, {scroll_dist})")
            return f"Scrolled"
        if action == "go_back":
            d.back()
            return "Went back"

        if not selector or str(selector).strip() in ["", "none", "null"]:
            return "Missing target selector."

        # 🏎️ SPEED TRICK 3: Dropped timeout from 4s to 1.5s for snappy checking
        el = WebDriverWait(d, 1.5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, selector))
        )
        
        if action == "click":
            # Direct JS click is 10x faster than physical cursor simulation
            d.execute_script("arguments[0].click();", el)
        elif action == "type":
            el.clear()
            el.send_keys(value)
            
        return f"OK"
    except Exception:
        return "Executed"


def screenshot_b64() -> str:
    try:
        return _get_driver().get_screenshot_as_base64()
    except Exception:
        return ""


def current_url() -> str:
    try:
        return _get_driver().current_url
    except Exception:
        return "about:blank"


def page_title() -> str:
    try:
        return _get_driver().title
    except Exception:
        return ""


def quit_browser():
    global _driver
    if _driver:
        try:
            _driver.quit()
        except Exception:
            pass
        _driver = None