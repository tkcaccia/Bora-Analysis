"""Create a review PDF from the manuscript Markdown when DOCX rendering is unavailable."""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak,
                                PageTemplate, Paragraph, Spacer, Table,
                                TableStyle)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from benchmark.runtime import BORA_ROOT


HERE = BORA_ROOT / 'docs'
SOURCE = HERE / "Bora_MIA_pilot_manuscript.md"
TARGET = HERE.parent / "output/pdf/Bora_MIA_working_preview.pdf"
TARGET.parent.mkdir(parents=True, exist_ok=True)

fontdir = Path("/usr/share/fonts/truetype/dejavu")
pdfmetrics.registerFont(TTFont("DejaVu", str(fontdir / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DejaVu-Bold", str(fontdir / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFontFamily("DejaVu", normal="DejaVu", bold="DejaVu-Bold")

styles = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("PaperTitle", fontName="DejaVu-Bold", fontSize=16, leading=21, alignment=TA_CENTER, spaceAfter=15),
    "h1": ParagraphStyle("PaperH1", fontName="DejaVu-Bold", fontSize=12, leading=16, spaceBefore=17, spaceAfter=8, keepWithNext=True),
    "h2": ParagraphStyle("PaperH2", fontName="DejaVu-Bold", fontSize=10.5, leading=14, spaceBefore=12, spaceAfter=6, keepWithNext=True),
    "body": ParagraphStyle("PaperBody", fontName="DejaVu", fontSize=9, leading=13.2, spaceAfter=8),
    "caption": ParagraphStyle("PaperCaption", fontName="DejaVu", fontSize=8.1, leading=11.5, spaceAfter=8),
    "tablecaption": ParagraphStyle("PaperTableCaption", fontName="DejaVu", fontSize=8.1, leading=11.5, spaceAfter=8, keepWithNext=True),
    "cell": ParagraphStyle("PaperCell", fontName="Times-Roman", fontSize=9, leading=12),
    "cellhead": ParagraphStyle("PaperCellHead", fontName="Times-Bold", fontSize=9, leading=12),
}


def rich(text: str) -> str:
    text = html.escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`(.+?)`", r"<font name='DejaVu'>\1</font>", text)
    return text.replace("–", "-").replace("—", "-")


def page(canvas, doc):
    canvas.saveState()
    canvas.setFont("DejaVu", 7.5)
    canvas.setFillColor(colors.HexColor("#536078"))
    canvas.drawString(47, 37, "Bora - Medical Image Analysis working preview")
    canvas.drawRightString(565, 37, f"{doc.page}")
    canvas.restoreState()


def figure_path(line: str) -> Path:
    m = re.search(r"Figure file: `([^`]+)`", line)
    if not m:
        raise ValueError(f"Missing figure path: {line}")
    return Path(m.group(1))


story = []
table_rows = []


def flush_table():
    if not table_rows:
        return
    count = len(table_rows[0])
    if count == 2:
        widths = [105, 390]
    elif count == 5:
        widths = [85, 125, 160, 55, 70]
    elif count == 8:
        widths = ([70, 46, 43, 62, 62, 42, 85, 85]
                  if table_rows[0][0] == 'Source' else
                  [44, 45, 75, 75, 45, 75, 75, 61])
    else:
        raise ValueError(f"Unexpected table with {count} columns")
    body = []
    for j, row in enumerate(table_rows):
        st = S["cellhead" if j == 0 else "cell"]
        body.append([Paragraph(rich(v), st) for v in row])
    t = Table(body, colWidths=widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("LINEABOVE", (0, 0), (-1, 0), .65, colors.HexColor("#666666")),
        ("LINEBELOW", (0, 0), (-1, 0), .5, colors.HexColor("#666666")),
        ("LINEBELOW", (0, -1), (-1, -1), .85, colors.HexColor("#666666")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("ALIGN", (3, 0), (-1, -1), "CENTER") if count == 5 else
            ("ALIGN", (0, 0), (-1, -1), "CENTER") if count == 8 else
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
    ]))
    story.extend([t, Spacer(1, 10)])
    table_rows.clear()


skip_pending = False
for line in SOURCE.read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    if line.startswith("|"):
        row = [v.strip() for v in line.strip("|").split("|")]
        if not all(set(v) <= set("-: ") for v in row):
            table_rows.append(row)
        continue
    flush_table()
    if line.startswith("## Submission items pending author input"):
        skip_pending = True
    if skip_pending or line.startswith("**Article draft") or line.startswith("**Affiliation verification:"):
        continue
    if line.startswith("**Authors, in requested order:**"):
        line = line.replace("Authors, in requested order:", "Authors:")
    if line.startswith("# "):
        story.append(Paragraph(rich(line[2:]), S["title"]))
    elif line.startswith("## "):
        story.append(Paragraph(rich(line[3:]), S["h1"]))
    elif line.startswith("### "):
        story.append(Paragraph(rich(line[4:]), S["h2"]))
    elif re.match(r"\*\*Figure [1-6]\.\*\*", line):
        if line.startswith("**Figure 4."):
            story.append(PageBreak())
        path = figure_path(line)
        caption = line.split(" Figure file:")[0]
        reader = ImageReader(str(path))
        w, h = reader.getSize()
        max_height = 350 if line.startswith("**Figure 1.") else 510 if line.startswith("**Figure 4.") else 270
        scale = min(495 / w, max_height / h)
        graphic = Image(str(path), width=w * scale, height=h * scale)
        graphic.hAlign = "CENTER"
        story.append(KeepTogether([graphic, Spacer(1, 5), Paragraph(rich(caption), S["caption"]), Spacer(1, 12)]))
    elif re.match(r"\*\*Table \d+\.\*\*", line):
        story.append(Paragraph(rich(line), S["tablecaption"]))
    else:
        story.append(Paragraph(rich(line), S["body"]))
flush_table()

doc = BaseDocTemplate(str(TARGET), pagesize=(612, 792),
                      leftMargin=47, rightMargin=47, topMargin=46, bottomMargin=55)
frame = Frame(47, 55, 518, 691, leftPadding=0, rightPadding=0,
              topPadding=0, bottomPadding=0)
doc.addPageTemplates(PageTemplate(id="paper", frames=frame, onPage=page))
doc.build(story)
print(TARGET)
