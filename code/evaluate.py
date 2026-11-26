"""Evaluacion: correr politicas sobre un periodo completo, benchmarks y metricas."""
from __future__ import annotations

import numpy as np
import pandas as pd

from code.env import BTCEnv
from code.reward import COSTO


def correr(env: BTCEnv, politica, semilla: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recorre el periodo completo (episodio=None) con la politica dada. Devuelve (bitacora, decisiones)."""
    obs, _ = env.reset(seed=semilla)
    decisiones, fin = [], False
    while not fin:
        t = env.t
        a = int(politica(obs))
        obs, r, term, trunc, info = env.step(a)
        decisiones.append((env.idx[t], a, r, info["duracion"]))
        fin = term or trunc
    bit = pd.DataFrame(env.bitacora)
    dec = pd.DataFrame(decisiones, columns=["ts", "accion", "recompensa", "duracion"])
    return bit, dec


def politica_modelo(modelo):
    return lambda obs: modelo.predict(obs, deterministic=True)[0]


def politica_aleatoria(semilla: int = 42):
    rng = np.random.default_rng(semilla)
    return lambda obs: int(rng.integers(3))


def curva_capital(bit: pd.DataFrame, indice: pd.DatetimeIndex, capital0: float = 100_000.0) -> pd.Series:
    """Capital por hora: cambia al cierre de cada operacion (la contribucion es terminal)."""
    eq = pd.Series(np.nan, index=indice)
    eq.iloc[0] = capital0
    if len(bit):
        cierres = bit.groupby("salida_ts")["pnl_neto"].sum().cumsum() + capital0
        eq.loc[cierres.index] = cierres.values
    return eq.ffill()


def btc_buy_hold(datos: pd.DataFrame, capital0: float = 100_000.0, costo: float = COSTO) -> pd.Series:
    """Compra al open de la primera barra y vende al cierre de la ultima, con costo en ambos lados."""
    eq = capital0 * (1 - costo) * datos["close"] / datos["open"].iloc[0]
    eq.iloc[-1] *= (1 - costo)
    return eq


def sp500_buy_hold(sp: pd.Series, ini, fin, capital0: float = 100_000.0, costo: float = COSTO) -> pd.Series:
    s = sp.loc[ini:fin]
    eq = capital0 * (1 - costo) * s / s.iloc[0]
    eq.iloc[-1] *= (1 - costo)
    return eq


def metricas(eq: pd.Series, bit: pd.DataFrame | None = None, periodos_anio: int = 365) -> dict:
    """Metricas sobre la curva de capital diaria."""
    d = eq.resample("1D").last().dropna() if isinstance(eq.index, pd.DatetimeIndex) and periodos_anio == 365 else eq
    r = d.pct_change().dropna()
    anios = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    total = eq.iloc[-1] / eq.iloc[0] - 1
    anual = (1 + total) ** (1 / anios) - 1
    vol = r.std() * np.sqrt(periodos_anio)
    neg = r[r < 0].std() * np.sqrt(periodos_anio)
    pico = eq.cummax()
    mdd = (eq / pico - 1).min()
    m = dict(retorno_total=total, retorno_anual=anual, volatilidad_anual=vol,
             sharpe=r.mean() / r.std() * np.sqrt(periodos_anio) if r.std() > 0 else 0.0,
             sortino=r.mean() * periodos_anio / neg if neg and neg > 0 else 0.0,
             max_drawdown=mdd, calmar=anual / abs(mdd) if mdd < 0 else 0.0)
    if bit is not None:
        n = len(bit)
        m.update(operaciones=n, hit_rate=(bit["pnl_neto"] > 0).mean() if n else 0.0,
                 duracion_media_h=bit["duracion"].mean() if n else 0.0,
                 rotacion_anual=2 * n / anios)  # veces el capital operado por anio (entrada + salida)
    return m


def resumen_trades(bit: pd.DataFrame) -> dict:
    if not len(bit):
        return dict(pct_ambiguas=0.0)
    return dict(pct_ambiguas=bit["ambigua"].mean(),
                pct_stop=(bit["razon"] == "stop").mean(), pct_tp=(bit["razon"] == "take_profit").mean(),
                pct_horizonte=(bit["razon"] == "horizonte").mean(), pct_largos=(bit["lado"] == 1).mean())
