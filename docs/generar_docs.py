"""Genera docs/reporte.pdf, docs/walkthrough.pdf y docs/presentacion.pdf a partir de artifacts/.

Requiere python-docx y Word (para exportar a PDF) en Windows. La presentacion se arma con matplotlib.
Uso: python docs/generar_docs.py
"""
import json
import os
import sys

import docx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx.enum.section import WD_ORIENT
from docx.shared import Cm, Pt
from matplotlib.backends.backend_pdf import PdfPages

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(RAIZ, "artifacts")
DOCS = os.path.join(RAIZ, "docs")
sys.path.insert(0, os.path.join(RAIZ, "docs"))
import textos  # noqa: E402  (redaccion del reporte, separada del formato)


def leer(n, **kw):
    return pd.read_csv(os.path.join(ART, n), **kw)


def pct(x):
    return f"{100 * x:.1f}%"


def tabla(d, df, fmt=None, tam=8):
    fmt = fmt or {}
    t = d.add_table(rows=1, cols=len(df.columns))
    t.style = "Table Grid"
    for c, h in zip(t.rows[0].cells, df.columns):
        c.text = str(h)
    for _, fila in df.iterrows():
        cel = t.add_row().cells
        for c, (col, v) in zip(cel, fila.items()):
            c.text = fmt[col](v) if col in fmt and pd.notna(v) else ("" if pd.isna(v) else str(v))
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(tam)
    d.add_paragraph()


def documento():
    d = docx.Document()
    s = d.styles["Normal"]
    s.font.name, s.font.size = "Calibri", Pt(10.5)
    for sec in d.sections:
        sec.left_margin = sec.right_margin = Cm(2)
        sec.top_margin = sec.bottom_margin = Cm(2)
    return d


def bloque(d, items):
    """items: lista de ('h1'|'h2'|'p'|'b'|'img'|'tabla', contenido)."""
    for tipo, x in items:
        if tipo == "h1":
            d.add_heading(x, 1)
        elif tipo == "h2":
            d.add_heading(x, 2)
        elif tipo == "p":
            d.add_paragraph(x)
        elif tipo == "b":
            d.add_paragraph(x, style="List Bullet")
        elif tipo == "img":
            ruta, ancho = x
            if os.path.exists(ruta):
                d.add_picture(ruta, width=Cm(ancho))
        elif tipo == "tabla":
            tabla(d, *x) if isinstance(x, tuple) else tabla(d, x)


def a_pdf(docx_path):
    import win32com.client
    w = win32com.client.Dispatch("Word.Application")
    w.Visible = False
    try:
        doc = w.Documents.Open(docx_path, False, True)
        doc.ExportAsFixedFormat(docx_path[:-5] + ".pdf", 17)
        paginas = doc.ComputeStatistics(2)
        doc.Close(False)
    finally:
        w.Quit()
    return paginas


def main():
    datos = textos.cargar(ART)
    for nombre, constructor in [("reporte", textos.reporte), ("walkthrough", textos.walkthrough)]:
        d = documento()
        bloque(d, constructor(datos, ART))
        ruta = os.path.join(DOCS, f"{nombre}.docx")
        d.save(ruta)
        print(nombre, a_pdf(ruta), "paginas")
        os.remove(ruta)
    if "--presentacion" not in sys.argv:  # la presentacion la arma el equipo por separado
        return
    with PdfPages(os.path.join(DOCS, "presentacion_borrador.pdf")) as pdf:
        n = 0
        for lam in textos.presentacion(datos, ART):
            fig = plt.figure(figsize=(13.33, 7.5))
            fig.text(0.05, 0.92, lam["titulo"], fontsize=26, weight="bold")
            if lam.get("img") and os.path.exists(lam["img"]):
                ax = fig.add_axes([0.02, 0.04, 0.6 if lam.get("texto") else 0.9, 0.8])
                ax.imshow(plt.imread(lam["img"]))
                ax.axis("off")
            if lam.get("texto"):
                x0 = 0.62 if lam.get("img") else 0.06
                fig.text(x0, 0.82, "\n\n".join(lam["texto"]), fontsize=15 if lam.get("img") else 18,
                         va="top", wrap=True)
            pdf.savefig(fig)
            plt.close(fig)
            n += 1
    print("presentacion", n, "laminas")


if __name__ == "__main__":
    main()
