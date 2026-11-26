"""Graficas del pipeline. Todas con titulo, ejes etiquetados y leyenda."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def curva_entrenamiento(curvas: dict, ruta: str, fold: int):
    """curvas: {etiqueta: (pasos, recompensas)}. Anota el maximo de la media movil de cada curva."""
    fig, ax = plt.subplots(figsize=(10, 5))
    for etq, (x, y) in curvas.items():
        if len(y) == 0:
            continue
        mm = pd.Series(y).rolling(max(len(y) // 20, 1), min_periods=1).mean()
        ax.plot(x, mm, label=etq)
        k = int(mm.idxmax())
        ax.annotate(f"máx {mm[k]:.2f}", (x[k], mm[k]), fontsize=7, xytext=(5, 5), textcoords="offset points")
    ax.axhline(0, color="gray", lw=0.8, ls="--", label="recompensa 0 (no operar)")
    ax.set(title=f"Fold {fold}: recompensa por episodio (media móvil) vs. pasos de entrenamiento",
           xlabel="pasos de entrenamiento (decisiones)", ylabel="recompensa acumulada por episodio")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def curvas_capital(series: dict, ruta: str, titulo: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    for etq, s in series.items():
        ax.plot(s.index, s.values / s.iloc[0], label=etq)
    ax.set(title=titulo, xlabel="fecha", ylabel="capital (base 1)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def periodo_drawdown(precio: pd.Series, eq: pd.Series, bit: pd.DataFrame, ruta: str, titulo: str):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    a1.plot(precio.index, precio.values, color="black", lw=0.8, label="BTCUSDT")
    for lado, color, et in [(1, "green", "entrada larga"), (-1, "red", "entrada corta")]:
        b = bit[bit["lado"] == lado]
        a1.scatter(b["entrada_ts"], b["precio_entrada"], marker="^" if lado == 1 else "v", color=color, s=25, label=et)
    a1.set(title=titulo, ylabel="precio (USDT)")
    a1.legend(fontsize=8)
    a2.plot(eq.index, eq.values, label="capital del agente")
    a2.set(xlabel="fecha", ylabel="capital (USDT)")
    a2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
