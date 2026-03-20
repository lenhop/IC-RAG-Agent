"""
Fetch a URL and save the response as PDF.

Tries methods in order: Playwright (best) -> WeasyPrint -> requests + fpdf2 (fallback).
Note: Amazon Seller Central may require authentication; if the page returns
"Server Busy" or a login form, you may need to use a browser session/cookies.

Usage:
    python fetch_url_to_pdf.py
    python fetch_url_to_pdf.py --url "https://example.com" --output "output.pdf"
"""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = "https://sellercentral.amazon.com/help/hub/reference/GHN4W67N7GNKMCMB"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "documents" / "sales_platform" / "amazon" / "amazon_seller_help_GHN4W67N7GNKMCMB.pdf"


def _fetch_with_playwright(url: str, output_path: Path) -> bool:
    """Use Playwright to render page and save PDF."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return False

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="networkidle", timeout=30000)
        page.pdf(path=str(output_path))
        browser.close()
    return True


def _fetch_with_weasyprint(url: str, output_path: Path) -> bool:
    """Use WeasyPrint to render page and save PDF."""
    try:
        from weasyprint import HTML
    except (ImportError, OSError):
        return False

    HTML(url=url).write_pdf(str(output_path))
    return True


def _fetch_with_requests_fpdf(url: str, output_path: Path) -> bool:
    """Fallback: fetch HTML with requests, extract text, write PDF with fpdf2."""
    try:
        import requests
        from bs4 import BeautifulSoup
        from fpdf import FPDF
    except ImportError:
        return False

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    if not text:
        text = f"Fetched from {url}\n\n(No text content extracted)"
    text = text[:50000]  # Limit PDF size

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.set_margins(15, 15, 15)
    for line in text.split("\n"):
        line = line.strip().replace("\r", "")[:180] or " "
        try:
            pdf.multi_cell(0, 6, line)
        except Exception:
            pdf.multi_cell(0, 6, " ")  # Skip problematic lines
    pdf.output(str(output_path))
    return True


def fetch_url_to_pdf(url: str, output_path: Path) -> Path:
    """
    Fetch URL and save response as PDF.

    Tries Playwright, then WeasyPrint, then requests+fpdf2.

    Args:
        url: URL to fetch.
        output_path: Path to save PDF.

    Returns:
        Path to saved PDF file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for name, fn in [
        ("playwright", _fetch_with_playwright),
        ("weasyprint", _fetch_with_weasyprint),
        ("requests+fpdf2", _fetch_with_requests_fpdf),
    ]:
        try:
            if fn(url, output_path):
                print(f"Saved using {name}")
                return output_path
        except Exception as e:
            print(f"  {name} failed: {e}")

    raise RuntimeError(
        "All PDF methods failed. Install: pip install playwright playwright install chromium"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch URL and save as PDF")
    parser.add_argument(
        "--url",
        type=str,
        default=DEFAULT_URL,
        help="URL to fetch",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output PDF path",
    )
    args = parser.parse_args()

    output = fetch_url_to_pdf(args.url, args.output)
    print(f"Saved to: {output}")


if __name__ == "__main__":
    main()
