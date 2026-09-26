"""Render the checked-in evaluation report as a self-contained, printable HTML page.

Run from any directory with a Python environment containing markdown-it-py:
    .venv/bin/python eval_out/render_evaluation_report.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from markdown_it import MarkdownIt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs/comprehensive_evaluation_report.md"
OUTPUT = SOURCE.with_suffix(".html")

STYLE = """
:root { --ink:#192d3b; --muted:#587080; --accent:#007c83; --line:#dce5e9;
  --paper:#fff; --canvas:#f1f5f7; --nav:#142a38; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; scroll-padding-top:26px; }
body { margin:0; color:var(--ink); background:var(--canvas);
  font:16px/1.68 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
a { color:var(--accent); text-underline-offset:3px; overflow-wrap:anywhere; }
a:hover { text-decoration-thickness:2px; }
.sidebar { position:fixed; inset:0 auto 0 0; width:275px; overflow:auto;
  background:var(--nav); color:#edf4f7; padding:30px 22px; }
.brand { font-size:25px; font-weight:750; letter-spacing:-.8px; }
.eyebrow { font-size:11px; text-transform:uppercase; letter-spacing:1.8px;
  color:#81cbd0; margin:3px 0 26px; }
.sidebar nav a { display:block; padding:7px 0; font-size:12px; line-height:1.5;
  text-decoration:none; color:#d5e1e8; }
.sidebar nav a:hover { color:#fff; }
.sidebar .tools { border-top:1px solid #45606c; margin-top:24px; padding-top:20px; }
.tools a,.tools button { font:inherit; font-size:12px; color:#fff; }
.tools button { cursor:pointer; background:#24616b; border:0; border-radius:5px;
  padding:8px 12px; margin:0 10px 10px 0; }
main { margin-left:275px; padding:36px 4vw 70px; }
article { max-width:1230px; margin:auto; background:var(--paper);
  padding:45px 54px; border:1px solid var(--line); border-radius:10px; }
h1 { font-size:40px; line-height:1.16; letter-spacing:-1.4px; max-width:850px;
  margin:0 0 23px; }
h2 { font-size:26px; line-height:1.3; margin:62px 0 22px;
  padding-top:22px; border-top:2px solid var(--line); letter-spacing:-.5px; }
h3 { font-size:19px; line-height:1.4; margin:32px 0 12px; }
p { margin:14px 0; }
strong { font-weight:700; }
ul,ol { padding-left:24px; }
li { margin:9px 0; }
code { font:12.5px/1.6 ui-monospace,SFMono-Regular,Consolas,monospace;
  background:#eef3f5; padding:2px 4px; border-radius:3px; overflow-wrap:anywhere; }
.table-wrap { width:100%; overflow-x:auto; margin:23px 0 27px;
  border:1px solid var(--line); border-radius:6px; }
table { width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.5;
  font-variant-numeric:tabular-nums; }
th { background:#e9f1f4; font-weight:700; color:#193b4a; }
th,td { padding:10px 11px; text-align:left; border-bottom:1px solid var(--line);
  vertical-align:top; }
td[style*="right"],th[style*="right"] { white-space:nowrap; }
tbody tr:nth-child(even) { background:#f8fafb; }
tbody tr:last-child td { border-bottom:0; }
tbody tr:hover { background:#edf7f6; }
.footer { margin-top:45px; padding-top:18px; border-top:1px solid var(--line);
  font-size:12px; color:var(--muted); }
@media (max-width:1100px) {
  .sidebar { width:235px; padding:24px 18px; }
  main { margin-left:235px; padding:22px; }
  article { padding:30px; }
}
@media (max-width:800px) {
  .sidebar { position:static; width:auto; padding:20px 24px; }
  .sidebar nav { columns:2; column-gap:24px; }
  .sidebar .eyebrow { margin-bottom:14px; }
  .sidebar .tools { margin-top:14px; padding-top:12px; }
  main { margin-left:0; padding:15px; }
  article { padding:26px 20px; }
  h1 { font-size:31px; }
  h2 { font-size:23px; }
}
@media print {
  @page { size:A4; margin:14mm 12mm; }
  body { background:white; font-size:9.5pt; line-height:1.45; }
  .sidebar { display:none; }
  main { margin:0; padding:0; }
  article { padding:0; border:0; border-radius:0; max-width:none; }
  h1 { font-size:27pt; }
  h2 { font-size:17pt; margin-top:26pt; padding-top:12pt; }
  h3 { font-size:12pt; margin-top:17pt; }
  h1,h2,h3 { break-after:avoid; }
  .table-wrap { overflow:visible; border-radius:0; margin:12pt 0; }
  table { font-size:7.5pt; }
  th,td { padding:5pt 4pt; }
  thead { display:table-header-group; }
  tr { break-inside:avoid; }
  a { color:inherit; text-decoration:none; }
  code { font-size:8pt; }
  li { margin:5pt 0; }
}
"""


def render() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    # Fail early rather than shipping a report with broken source citations.
    links = re.findall(r"\]\(([^)]+)\)", source)
    for link in links:
        if link.startswith(("http:", "https:", "#")):
            continue
        target = SOURCE.parent / link.split("#", 1)[0]
        if not target.exists():
            raise FileNotFoundError(f"Report citation does not exist: {link}")

    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = md.parse(source)
    toc = []
    used = set()
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        title = tokens[index + 1].content
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        anchor = slug
        suffix = 2
        while anchor in used:
            anchor = f"{slug}-{suffix}"
            suffix += 1
        used.add(anchor)
        token.attrSet("id", anchor)
        if token.tag == "h2":
            toc.append(f'<a href="#{anchor}">{html.escape(title)}</a>')

    body = md.renderer.render(tokens, md.options, {})
    body = body.replace("<table>", '<div class="table-wrap"><table>')
    body = body.replace("</table>", "</table></div>")
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="A source-linked synthesis of GeoAE reconstruction, clustering, semantic probes, causal editing, number control and corrected audits.">
<title>GeoAE — Comprehensive Evaluation Report</title>
<style>{STYLE}</style>
</head>
<body>
<aside class="sidebar">
<div class="brand">GeoAE</div>
<div class="eyebrow">Evaluation report · September 2026</div>
<nav aria-label="Report contents">{''.join(toc)}</nav>
<div class="tools"><button onclick="window.print()">Print / save PDF</button>
<a href="{SOURCE.name}">Markdown source</a></div>
</aside>
<main><article>
{body}
<div class="footer">Prepared from saved repository artifacts and reconciled Claude notes.
Snapshot: 23 September 2026. This page uses no external fonts, scripts or services.</div>
</article></main>
</body></html>
"""
    OUTPUT.write_text(document, encoding="utf-8")
    print(f"Rendered {OUTPUT.relative_to(ROOT)}")
    print(f"Checked {len(links)} source links; {len(toc)} navigable sections.")


if __name__ == "__main__":
    render()
