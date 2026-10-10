#!/usr/bin/env python3
"""Render a DOCX body (paragraphs and tables) as readable Markdown."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path
from xml.etree import ElementTree

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS = {"w": W_NS}


def text_in(parent: ElementTree.Element) -> str:
    return "".join(parent.itertext()).replace("\xa0", " ").strip()


def render_paragraph(paragraph: ElementTree.Element) -> str:
    parts: list[str] = []
    for node in paragraph:
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "r":
            parts.append("".join(node.itertext()))
        elif tag == "hyperlink":
            parts.append("".join(node.itertext()))
    return "".join(parts).replace("\xa0", " ").strip()


def render_table(table: ElementTree.Element) -> list[str]:
    rows: list[list[str]] = []
    for row in table.findall("w:tr", NS):
        cells = []
        for cell in row.findall("w:tc", NS):
            value = " ".join(
                rendered for paragraph in cell.findall("w:p", NS)
                if (rendered := render_paragraph(paragraph))
            )
            cells.append(value.replace("|", "\\|"))
        if cells:
            rows.append(cells)
    if not rows:
        return []
    width = max(map(len, rows))
    rows = [row + [""] * (width - len(row)) for row in rows]
    lines = ["| " + " | ".join(rows[0]) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    return lines


def render_docx(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ElementTree.fromstring(archive.read("word/document.xml"))
    body = root.find("w:body", NS)
    if body is None:
        raise ValueError(f"{path} has no document body")
    output: list[str] = []
    for child in body:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            value = render_paragraph(child)
            if value:
                output.append(value)
        elif tag == "tbl":
            output.extend(render_table(child))
            output.append("")
    return "\n\n".join(output).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    args = parser.parse_args()
    rendered = render_docx(args.input)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Rendered {len(rendered.splitlines())} lines to {args.output}")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
