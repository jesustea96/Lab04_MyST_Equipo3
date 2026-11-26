"""Carga de BTCUSDT horario, features causales, folds walk-forward y normalizacion por fold."""
from __future__ import annotations

import numpy as np
import pandas as pd

RAW = "data/Binance_BTCUSDT_1h.csv"

# Folds de ventana expansiva (seccion 3.6 del enunciado): (train_fin, val_ini, val_fin, test_ini, test_fin)
FOLDS = {
    1: ("2021-12-31 23:00", "2022-01-01", "2022-12-31 23:00", "2023-01-01", "2023-12-31 23:00"),
    2: ("2022-12-31 23:00", "2023-01-01", "2023-12-31 23:00", "2024-01-01", "2024-12-31 23:00"),
    3: ("2023-12-31 23:00", "2024-01-01", "2024-12-31 23:00", "2025-01-01", "2025-12-31 23:00"),
    4: ("2024-12-31 23:00", "2025-01-01", "2025-12-31 23:00", "2026-01-01", "2026-04-26 23:00"),
}
INICIO = "2017-08-17 04:00"

# Familias de variables (para la ablacion). Todas se calculan con informacion hasta el cierre de t-1.
FAMILIAS = {
    "momentum": ["ret_1", "ret_6", "ret_24", "ret_168", "dist_ma168", "rsi_14"],
    "volatilidad": ["vol_24", "vol_168", "parkinson_24", "vol_ratio"],
    "volumen": ["volz_24", "tradez_24"],
    "calendario": ["hora_sin", "hora_cos"],
    "creencia": ["sigma_ewma", "tend_168", "tend_t"],
}
FEATURES = [f for fam in FAMILIAS.values() for f in fam]


def cargar(ruta: str = RAW) -> pd.DataFrame:
    """Lee el csv de CryptoDataDownload (viene en orden descendente y con una linea de encabezado extra)."""
    df = pd.read_csv(ruta, skiprows=1)
    df["Date"] = pd.to_datetime(df["Date"], format="mixed")
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close",
                            "Volume BTC": "vol_btc", "tradecount": "trades"})
    df = df.sort_values("Date").drop_duplicates("Date").set_index("Date")
    return df[["open", "high", "low", "close", "vol_btc", "trades"]].astype(float)


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _pendiente(y: pd.Series, n: int) -> tuple[pd.Series, pd.Series]:
    """Pendiente de una regresion lineal movil de log-precio y su estadistico t (la incertidumbre de la creencia)."""
    x = np.arange(n) - (n - 1) / 2
    sxx = (x ** 2).sum()
    b = y.rolling(n).apply(lambda v: (x * (v - v.mean())).sum() / sxx, raw=True)
    resid_var = y.rolling(n).var() * (n - 1) - b ** 2 * sxx
    se = np.sqrt(np.maximum(resid_var, 1e-18) / (n - 2) / sxx)
    return b, b / se


def features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula las variables de informacion y de creencia.

    Todas se desplazan una barra (shift(1)): la fila t solo contiene datos hasta el cierre de t-1,
    y la decision se ejecuta al open de t. Asi no hay look-ahead.
    """
    c = np.log(df["close"])
    r = c.diff()
    f = pd.DataFrame(index=df.index)
    for k in (1, 6, 24, 168):
        f[f"ret_{k}"] = c - c.shift(k)
    f["dist_ma168"] = c - c.rolling(168).mean()
    f["rsi_14"] = (_rsi(df["close"]) - 50) / 50
    f["vol_24"] = r.rolling(24).std()
    f["vol_168"] = r.rolling(168).std()
    f["parkinson_24"] = np.sqrt((np.log(df["high"] / df["low"]) ** 2).rolling(24).mean() / (4 * np.log(2)))
    f["vol_ratio"] = f["vol_24"] / f["vol_168"]
    lv = np.log1p(df["vol_btc"])
    f["volz_24"] = (lv - lv.rolling(168).mean()) / lv.rolling(168).std()
    lt = np.log1p(df["trades"])
    f["tradez_24"] = (lt - lt.rolling(168).mean()) / lt.rolling(168).std()
    # La hora de la decision es conocida en t, por eso no se desplaza
    h = df.index.hour
    f["sigma_ewma"] = np.sqrt((r ** 2).ewm(halflife=24, adjust=False).mean())
    f["tend_168"], f["tend_t"] = _pendiente(c, 168)
    f = f.shift(1)
    f["sigma_ewma_raw"] = f["sigma_ewma"]  # sin normalizar: la usa el gestor para fijar stop y TP
    f["hora_sin"] = np.sin(2 * np.pi * h / 24)
    f["hora_cos"] = np.cos(2 * np.pi * h / 24)
    return f


def preparar(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Precios + features, sin las primeras filas con NaN (ventanas de 168 h)."""
    df = cargar() if df is None else df
    out = df.join(features(df))
    return out.dropna()


def particion(datos: pd.DataFrame, fold: int) -> dict[str, pd.DataFrame]:
    tr_fin, va_ini, va_fin, te_ini, te_fin = FOLDS[fold]
    return {"train": datos.loc[INICIO:tr_fin], "val": datos.loc[va_ini:va_fin], "test": datos.loc[te_ini:te_fin]}


class Normalizador:
    """z-score con media y desviacion del entrenamiento del fold; se aplica sin cambios a validacion y prueba."""

    def __init__(self, cols: list[str], clip: float = 5.0):
        self.cols, self.clip = cols, clip

    def fit(self, df: pd.DataFrame) -> "Normalizador":
        self.mu = df[self.cols].mean()
        self.sd = df[self.cols].std().replace(0, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.cols] = ((df[self.cols] - self.mu) / self.sd).clip(-self.clip, self.clip)
        return out
