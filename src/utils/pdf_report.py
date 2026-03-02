"""
PDF Report Generator -- Creates professional PDF reports with embedded charts.
Uses fpdf2 with Unicode font support for proper rendering of Greek letters,
mathematical symbols, and statistical notation (beta, R^2, etc.).
"""

import os
import re
from pathlib import Path

from fpdf import FPDF


# ─── Font discovery: find a Unicode-capable TTF on the system ────────────────

_UNICODE_FONT_CANDIDATES = [
    # Windows system fonts with good Unicode/Greek coverage
    r"C:\Windows\Fonts\DejaVuSans.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\times.ttf",
    # Linux
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    # macOS
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]

_UNICODE_FONT_BOLD_CANDIDATES = [
    r"C:\Windows\Fonts\DejaVuSans-Bold.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    r"C:\Windows\Fonts\timesbd.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]

_UNICODE_FONT_ITALIC_CANDIDATES = [
    r"C:\Windows\Fonts\DejaVuSans-Oblique.ttf",
    r"C:\Windows\Fonts\calibrii.ttf",
    r"C:\Windows\Fonts\ariali.ttf",
    r"C:\Windows\Fonts\segoeuii.ttf",
    r"C:\Windows\Fonts\timesi.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
]


def _find_font(candidates: list[str]) -> str | None:
    """Find the first available font file from the candidate list."""
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


# ─── ASCII fallback (only used if no Unicode font found) ─────────────────────

_UNICODE_REPLACEMENTS = {
    "\u03b2": "beta", "\u03b1": "alpha", "\u03b5": "epsilon",
    "\u03c3": "sigma", "\u03bc": "mu", "\u03bb": "lambda",
    "\u03b4": "delta", "\u03b3": "gamma", "\u03a3": "Sigma",
    "\u0394": "Delta", "\u03c0": "pi", "\u03c7": "chi",
    "\u03c1": "rho", "\u03c4": "tau", "\u03b8": "theta",
    "\u03c6": "phi", "\u03c8": "psi", "\u03c9": "omega",
    "\u03b7": "eta", "\u03b6": "zeta", "\u03bd": "nu",
    "\u00b2": "^2", "\u00b3": "^3", "\u00b9": "^1",
    "\u2080": "0", "\u2081": "1", "\u2082": "2",
    "\u2083": "3", "\u2084": "4", "\u2085": "5",
    "\u2264": "<=", "\u2265": ">=", "\u2260": "!=",
    "\u00b1": "+/-", "\u00d7": "x", "\u00f7": "/",
    "\u2192": "->", "\u2190": "<-", "\u221e": "inf",
    "\u221a": "sqrt", "\u2211": "sum", "\u2248": "~=",
    "\u2013": "-", "\u2014": "--",
    "\u2018": "'", "\u2019": "'",
    "\u201c": '"', "\u201d": '"',
    "\u2022": "-", "\u2026": "...", "\u00b0": " deg",
}


def _sanitize_text(text: str) -> str:
    """Replace Unicode characters with ASCII equivalents (fallback only)."""
    for uc, asc in _UNICODE_REPLACEMENTS.items():
        text = text.replace(uc, asc)
    text = text.encode("ascii", errors="replace").decode("ascii")
    return text


# ─── PDF class ───────────────────────────────────────────────────────────────

class StatEnginePDF(FPDF):
    """Custom PDF with Unicode font support, styled headers, and chart embedding."""

    def __init__(self, title: str = "Stat Engine Report"):
        super().__init__()
        self.report_title = title
        self.set_auto_page_break(auto=True, margin=20)

        # Try to register a Unicode font
        self._has_unicode = False
        self._font_family = "Helvetica"  # fallback

        font_regular = _find_font(_UNICODE_FONT_CANDIDATES)
        if font_regular:
            try:
                self.add_font("UniFont", "", font_regular)
                # Try bold variant
                font_bold = _find_font(_UNICODE_FONT_BOLD_CANDIDATES)
                if font_bold:
                    self.add_font("UniFont", "B", font_bold)
                else:
                    self.add_font("UniFont", "B", font_regular)
                # Try italic variant
                font_italic = _find_font(_UNICODE_FONT_ITALIC_CANDIDATES)
                if font_italic:
                    self.add_font("UniFont", "I", font_italic)
                else:
                    self.add_font("UniFont", "I", font_regular)
                self._has_unicode = True
                self._font_family = "UniFont"
            except Exception:
                pass  # Fall back to Helvetica

    def _safe_text(self, text: str) -> str:
        """Sanitize text only if using non-Unicode font."""
        if self._has_unicode:
            return text
        return _sanitize_text(text)

    def _set_font(self, style: str = "", size: int = 10):
        """Set font using the best available family."""
        self.set_font(self._font_family, style, size)

    def header(self):
        if self.page_no() > 1:
            self._set_font("I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, self._safe_text(self.report_title), align="L")
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self._set_font("I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    # ── Cover page ────────────────────────────────────────────────────────

    def _add_cover_page(self, title: str, subtitle: str = ""):
        self.add_page()

        # Dark header band
        self.set_fill_color(35, 39, 65)
        self.rect(0, 0, 210, 105, "F")

        # Accent stripe
        self.set_fill_color(100, 126, 234)
        self.rect(0, 100, 210, 5, "F")

        # Title
        self.set_y(28)
        self._set_font("B", 28)
        self.set_text_color(255, 255, 255)
        self.cell(0, 15, self._safe_text(title), align="C",
                  new_x="LMARGIN", new_y="NEXT")

        # Subtitle
        if subtitle:
            self.ln(3)
            self._set_font("", 14)
            self.set_text_color(180, 190, 230)
            self.cell(0, 10, self._safe_text(subtitle), align="C",
                      new_x="LMARGIN", new_y="NEXT")

        # Decorative line
        self.ln(10)
        self.set_draw_color(100, 126, 234)
        self.set_line_width(0.8)
        self.line(65, self.get_y(), 145, self.get_y())

        # Meta info
        self.set_y(118)
        self.set_text_color(80, 80, 80)
        self._set_font("", 11)

        from datetime import datetime
        self.cell(0, 8,
                  f"Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}",
                  align="C", new_x="LMARGIN", new_y="NEXT")
        self.cell(0, 8, "Powered by Stat Engine Agent",
                  align="C", new_x="LMARGIN", new_y="NEXT")

        # Unicode indicator
        if self._has_unicode:
            self.ln(5)
            self._set_font("I", 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, "Enhanced PDF with full Unicode symbol support",
                      align="C", new_x="LMARGIN", new_y="NEXT")

    # ── Section headers ───────────────────────────────────────────────────

    def _add_section_header(self, text: str, level: int = 1):
        text = self._safe_text(text)
        self.ln(4)

        if level == 1:
            # Blue background band for H1
            self.set_fill_color(240, 242, 255)
            self.set_draw_color(100, 126, 234)
            self._set_font("B", 15)
            self.set_text_color(35, 39, 65)
            self.cell(0, 10, f"  {text}", new_x="LMARGIN", new_y="NEXT",
                      fill=True, border="B")
            self.ln(4)
        elif level == 2:
            self._set_font("B", 13)
            self.set_text_color(60, 65, 110)
            self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT")
            # Subtle underline
            self.set_draw_color(180, 190, 220)
            self.set_line_width(0.3)
            self.line(self.l_margin, self.get_y(),
                      self.l_margin + self.get_string_width(text) + 5, self.get_y())
            self.ln(3)
        else:
            self._set_font("B", 11)
            self.set_text_color(80, 85, 120)
            self.cell(0, 7, text, new_x="LMARGIN", new_y="NEXT")
            self.ln(1)

    # ── Body text ─────────────────────────────────────────────────────────

    def _add_body_text(self, text: str):
        text = self._safe_text(text)
        self._set_font("", 10)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def _add_bullet(self, text: str, indent: int = 0):
        text = self._safe_text(text)
        self._set_font("", 10)
        self.set_text_color(40, 40, 40)
        x = self.l_margin + indent * 5 + 2
        self.set_x(x)

        # Use a proper bullet if Unicode is available, dash otherwise
        bullet = "\u2022 " if self._has_unicode else "- "
        self.cell(6, 5.5, bullet, new_x="END")
        self.multi_cell(0, 5.5, text, new_x="LMARGIN", new_y="NEXT")

    # ── Stat highlight box ────────────────────────────────────────────────

    def _add_stat_box(self, label: str, value: str):
        """Add a highlighted statistic (e.g., R² = 0.85)."""
        label = self._safe_text(label)
        value = self._safe_text(value)
        self.set_fill_color(245, 247, 255)
        self.set_draw_color(100, 126, 234)
        self._set_font("B", 10)
        self.set_text_color(35, 39, 65)
        self.cell(50, 8, f"  {label}: ", new_x="END", fill=True, border="L")
        self._set_font("", 10)
        self.set_text_color(60, 60, 60)
        self.cell(0, 8, value, new_x="LMARGIN", new_y="NEXT", fill=True)

    # ── Chart embedding ───────────────────────────────────────────────────

    def _add_chart_image(self, image_path: str, caption: str = ""):
        if not os.path.exists(image_path):
            return

        if self.get_y() > 190:
            self.add_page()

        self.ln(3)

        page_width = 210 - self.l_margin - self.r_margin
        img_width = min(page_width, 170)

        try:
            self.image(image_path, x=(210 - img_width) / 2, w=img_width)
        except Exception:
            self._add_body_text(
                f"[Chart could not be embedded: {Path(image_path).name}]"
            )
            return

        if caption:
            caption = self._safe_text(caption)
            self.ln(2)
            self._set_font("I", 9)
            self.set_text_color(100, 100, 110)
            self.cell(0, 6, caption, align="C",
                      new_x="LMARGIN", new_y="NEXT")
        self.ln(4)


# ─── Markdown parser ─────────────────────────────────────────────────────────

def _parse_markdown_sections(text: str) -> list[dict]:
    """Parse markdown text into sections with level, title, content."""
    text = re.sub(r"^```\w*\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)

    sections = []
    current = {"level": 0, "title": "", "content": ""}

    for line in text.split("\n"):
        header_match = re.match(r"^(#{1,4})\s+(.+)$", line)
        if header_match:
            if current["title"] or current["content"].strip():
                sections.append(current)
            level = len(header_match.group(1))
            current = {"level": level, "title": header_match.group(2),
                        "content": ""}
        else:
            current["content"] += line + "\n"

    if current["title"] or current["content"].strip():
        sections.append(current)

    return sections


def _get_chart_images(charts_dir: str) -> list[str]:
    """Get sorted list of chart PNG paths."""
    if not os.path.isdir(charts_dir):
        return []
    return [str(f) for f in sorted(Path(charts_dir).glob("*.png"))]


# ─── Main generator ──────────────────────────────────────────────────────────

def generate_pdf(report_text: str, charts_dir: str, output_path: str):
    """
    Generate a professional PDF report with embedded charts.

    Args:
        report_text: The markdown report text.
        charts_dir: Path to the directory containing chart PNGs.
        output_path: Where to save the generated PDF.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pdf = StatEnginePDF(title="Stat Engine Analysis Report")
    pdf.alias_nb_pages()

    # Cover page
    pdf._add_cover_page(
        "Statistical Analysis Report",
        "Automated Data Analysis & Insights",
    )

    sections = _parse_markdown_sections(report_text)
    chart_images = _get_chart_images(charts_dir)
    charts_inserted = False

    pdf.add_page()

    for section in sections:
        if section["title"]:
            pdf._add_section_header(section["title"], level=section["level"])

        content = section["content"].strip()
        if not content:
            continue

        for line in content.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Bullet points (various indent levels)
            if line.startswith("- ") or line.startswith("* "):
                bullet_text = line[2:].strip()
                bullet_text = re.sub(r"\*\*(.+?)\*\*", r"\1", bullet_text)
                pdf._add_bullet(bullet_text)
            elif line.startswith("  - ") or line.startswith("  * "):
                bullet_text = line[4:].strip()
                bullet_text = re.sub(r"\*\*(.+?)\*\*", r"\1", bullet_text)
                pdf._add_bullet(bullet_text, indent=2)
            elif line.startswith("    - ") or line.startswith("    * "):
                bullet_text = line[6:].strip()
                bullet_text = re.sub(r"\*\*(.+?)\*\*", r"\1", bullet_text)
                pdf._add_bullet(bullet_text, indent=4)
            else:
                # Strip markdown formatting
                clean = re.sub(r"\*\*(.+?)\*\*", r"\1", line)
                clean = re.sub(r"\*(.+?)\*", r"\1", clean)
                clean = re.sub(r"`(.+?)`", r"\1", clean)
                # Skip markdown table separators
                if re.match(r"^\|?[\s\-:|]+\|?$", clean):
                    continue
                pdf._add_body_text(clean)

        # Insert charts after relevant sections
        title_lower = section["title"].lower() if section["title"] else ""
        chart_keywords = [
            "visual", "chart", "plot", "figure", "residual",
            "regression", "model fit", "exploratory",
        ]
        if any(kw in title_lower for kw in chart_keywords):
            if chart_images and not charts_inserted:
                for img_path in chart_images:
                    caption = Path(img_path).stem.replace("_", " ").title()
                    caption = re.sub(r"^\d+\s*", "", caption)
                    pdf._add_chart_image(img_path, caption)
                charts_inserted = True

    # Append charts at the end if not already inserted
    if chart_images and not charts_inserted:
        pdf.add_page()
        pdf._add_section_header("Charts & Visualizations", level=1)
        for img_path in chart_images:
            caption = Path(img_path).stem.replace("_", " ").title()
            caption = re.sub(r"^\d+\s*", "", caption)
            pdf._add_chart_image(img_path, caption)

    pdf.output(output_path)
