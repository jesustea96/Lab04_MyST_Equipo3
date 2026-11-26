"""Analisis exploratorio (solo datos de entrenamiento) y analisis de variables por fold."""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from statsmodels.tsa.stattools import acf, adfuller, kpss

warnings.filterwarnings("ignore", module="statsmodels")  # avisos de API futura y de p-valor fuera de tabla

# Regimenes definidos por rango de fechas dentro del entrenamiento mas largo (hasta 2024)
REGIMENES = [
    ("Burbuja 2017", "2017-08-17", "2017-12-17"),
    ("Invierno cripto 2018", "2017-12-18", "2018-12-15"),
    ("Recuperación 2019", "2018-12-16", "2019-06-26"),
    ("Lateral y crash COVID", "2019-06-27", "2020-03-12"),
    ("Toro 2020-2021", "2020-03-13", "2021-11-10"),
    ("Bajista 2022 (Luna, FTX)", "2021-11-11", "2022-11-21"),
    ("Recuperación y ETF 2023-2024", "2022-11-22", "2024-12-31"),
]


def tabla_regimenes(df: pd.DataFrame, hasta: str = "2024-12-31 23:00") -> pd.DataFrame:
    filas = []
    for nombre, a, b in REGIMENES:
        s = df.loc[a:min(b, hasta), "close"]
        if len(s) < 48:
            continue
        r = np.log(s).diff().dropna()
        filas.append(dict(regimen=nombre, inicio=a, fin=b, retorno_total=s.iloc[-1] / s.iloc[0] - 1,
                          vol_anual=r.std() * np.sqrt(24 * 365), tendencia_diaria=r.mean() * 24,
                          max_drawdown=(s / s.cummax() - 1).min()))
    return pd.DataFrame(filas)


def pruebas_estacionariedad(df: pd.DataFrame) -> pd.DataFrame:
    lp = np.log(df["close"])
    r = lp.diff().dropna()
    filas = []
    for nombre, x in [("log precio", lp), ("retorno horario", r), ("|retorno| horario", r.abs())]:
        x = x.iloc[-20_000:]  # ultimas 20k horas del entrenamiento, para que KPSS sea manejable
        adf = adfuller(x, autolag="AIC")
        kp = kpss(x, regression="c", nlags="auto")
        filas.append(dict(serie=nombre, adf_stat=adf[0], adf_p=adf[1], kpss_stat=kp[0], kpss_p=kp[1]))
    return pd.DataFrame(filas)


def eda_pdf(df_train: pd.DataFrame, ruta: str) -> dict:
    lp = np.log(df_train["close"])
    r = lp.diff().dropna()
    reg = tabla_regimenes(df_train)
    est = pruebas_estacionariedad(df_train)
    ac_r = acf(r, nlags=48)
    ac_a = acf(r.abs(), nlags=48)
    with PdfPages(ruta) as pdf:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(lp.index, lp.values, color="black", lw=0.7, label="log precio")
        colores = plt.cm.tab10.colors
        for i, (n, a, b) in enumerate(REGIMENES):
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color=colores[i % 10], alpha=0.15, label=n)
        ax.set(title="BTCUSDT (entrenamiento 2017-08 a 2024-12): log precio y regímenes", xlabel="fecha",
               ylabel="log(precio)")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5))
        z = (r - r.mean()) / r.std()
        ax.hist(z.clip(-10, 10), bins=200, density=True, alpha=0.6, label="retornos horarios estandarizados")
        xs = np.linspace(-10, 10, 400)
        ax.plot(xs, stats.norm.pdf(xs), label="normal estándar")
        ax.set(title=f"Distribución de retornos horarios: curtosis en exceso {stats.kurtosis(r):.1f}, "
                     f"asimetría {stats.skew(r):.2f}", xlabel="retorno estandarizado", ylabel="densidad", yscale="log")
        ax.legend()
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 5))
        lags = np.arange(1, 49)
        banda = 1.96 / np.sqrt(len(r))
        ax.bar(lags - 0.2, ac_r[1:], width=0.4, label="ACF retornos")
        ax.bar(lags + 0.2, ac_a[1:], width=0.4, label="ACF |retornos|")
        ax.axhline(banda, ls="--", color="gray", label="banda 95%"); ax.axhline(-banda, ls="--", color="gray")
        ax.set(title="Autocorrelación: el signo casi no se predice, la volatilidad sí (y con ciclo de 24 h)",
               xlabel="rezago (horas)", ylabel="autocorrelación")
        ax.legend()
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        for titulo, tabla in [("Regímenes del entrenamiento", reg), ("Pruebas de estacionariedad (ADF: H0 raíz "
                                                                       "unitaria; KPSS: H0 estacionaria)", est)]:
            fig, ax = plt.subplots(figsize=(11, 4))
            ax.axis("off")
            t = tabla.copy()
            for c in t.select_dtypes("number"):
                t[c] = t[c].map(lambda v: f"{v:.4f}")
            tb = ax.table(cellText=t.values, colLabels=t.columns, loc="center")
            tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.4)
            ax.set_title(titulo)
            pdf.savefig(fig); plt.close(fig)
    return dict(regimenes=reg, estacionariedad=est, acf_r=ac_r, acf_abs=ac_a)


def analisis_variables(df_train: pd.DataFrame, cols: list[str], horizonte: int, semilla: int = 42) -> pd.DataFrame:
    """Relacion de cada variable con el objetivo (retorno log de las siguientes H horas desde el open de t).
    Solo entrenamiento. El objetivo usa datos futuros, pero solo para medir relevancia, nunca como entrada."""
    fwd = np.log(df_train["close"].shift(-(horizonte - 1)) / df_train["open"])
    d = df_train[cols].assign(y=fwd).dropna()
    mi = mutual_info_regression(d[cols].values, d["y"].values, random_state=semilla, n_neighbors=5)
    return pd.DataFrame(dict(variable=cols, info_mutua=mi,
                             pearson=[d[c].corr(d["y"]) for c in cols],
                             spearman=[d[c].corr(d["y"], method="spearman") for c in cols],
                             corr_abs_y=[d[c].corr(d["y"].abs(), method="spearman") for c in cols])
                        ).sort_values("info_mutua", ascending=False)
