"""Funciones de contribucion. La contribucion es terminal: se paga al cerrar la operacion (semi-Markov)."""
from __future__ import annotations

import numpy as np

COSTO = 0.00125  # 0.125% por lado


def retorno_neto(lado: int, p_entrada: float, p_salida: float, costo: float = COSTO) -> float:
    """Retorno simple neto de una operacion de tamano 1 (sin apalancamiento), con costo en la entrada y en la salida."""
    bruto = lado * (p_salida / p_entrada - 1.0)
    return (1 + bruto) * (1 - costo) ** 2 - 1


def contribucion(r_neto: float, sigma: float, horizonte: int, modo: str = "riesgo") -> float:
    """Recompensa que ve el agente.

    - "pnl": retorno neto en por ciento. Premia tomar riesgo en regimenes volatiles.
    - "riesgo": retorno neto dividido entre la volatilidad esperada de la operacion (sigma * sqrt(H)),
      truncado a +-5. Hace comparables las operaciones de 2018 y de 2024 sin introducir coeficientes de moldeo.
    """
    if modo == "pnl":
        return 100.0 * r_neto
    escala = max(sigma * np.sqrt(horizonte), 1e-4)
    return float(np.clip(r_neto / escala, -5.0, 5.0))
