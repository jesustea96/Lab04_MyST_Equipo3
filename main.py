"""Punto de entrada unico del pipeline del Lab 04.

Uso:  python main.py            (corrida completa, semillas fijas)
      python main.py --rapido   (prueba de humo: pocos pasos, solo fold 1)
"""
import os
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
if "code" in sys.modules and not getattr(sys.modules["code"], "__file__", "").startswith(RAIZ):
    del sys.modules["code"]  # 'code' tambien es un modulo de la biblioteca estandar

import argparse
import json
import platform
import time

import gymnasium
import numpy as np
import pandas as pd
import stable_baselines3
import torch

from code import data, eda, evaluate as ev, plots
from code.env import BTCEnv
from code.train import entrenar

ART = os.path.join(RAIZ, "artifacts")
SEMILLAS = [42, 43, 44]
# Rejilla de validacion: horizonte de tenencia x funcion de contribucion; gamma alternativo para la discusion
REJILLA = [dict(horizonte=h, modo_reward=m, gamma=0.0) for h in (12, 24) for m in ("riesgo", "pnl")]
REJILLA.append(dict(horizonte=24, modo_reward="riesgo", gamma=0.9))
FIJOS = dict(k_stop=2.0, k_tp=3.0, episodio=720)


def env_kw(cfg: dict, **extra) -> dict:
    kw = dict(FIJOS, horizonte=cfg["horizonte"], modo_reward=cfg["modo_reward"])
    kw.update(extra)
    return kw


def evaluar(modelo, df, cols, cfg, **extra):
    env = BTCEnv(df, cols, **env_kw(cfg, episodio=None, **extra))
    bit, dec = ev.correr(env, ev.politica_modelo(modelo))
    eq = ev.curva_capital(bit, df.index)
    return bit, dec, eq, ev.metricas(eq, bit)


def sp500(ruta=os.path.join(RAIZ, "data", "sp500.csv")) -> pd.Series:
    """Cierres diarios del S&P 500 (^GSPC), solo como benchmark. Se descargan una vez y se congelan en data/."""
    if not os.path.exists(ruta):
        import yfinance as yf
        s = yf.download("^GSPC", start="2021-12-01", end="2026-04-30", auto_adjust=True, progress=False)["Close"]
        s = s.squeeze()
        s.name = "close"
        s.to_csv(ruta)
    s = pd.read_csv(ruta, index_col=0, parse_dates=True).squeeze("columns")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    ap.add_argument("--pasos", type=int, default=60_000)
    args = ap.parse_args()
    pasos = 3_000 if args.rapido else args.pasos
    folds = [1] if args.rapido else [1, 2, 3, 4]
    os.makedirs(ART, exist_ok=True)
    t_ini = time.time()

    meta = dict(fecha=time.strftime("%Y-%m-%d %H:%M"), python=platform.python_version(),
                stable_baselines3=stable_baselines3.__version__, gymnasium=gymnasium.__version__,
                torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__, semillas=SEMILLAS,
                pasos=pasos, rejilla=REJILLA, fijos=FIJOS, costo_por_lado=0.00125)

    datos = data.preparar()
    crudo = data.cargar()
    sp = sp500()
    eda.eda_pdf(crudo.loc[data.INICIO:"2024-12-31 23:00"], os.path.join(ART, "btcusdt_eda.pdf"))
    eda.tabla_regimenes(crudo).to_csv(os.path.join(ART, "regimenes.csv"), index=False)
    eda.pruebas_estacionariedad(crudo.loc[data.INICIO:"2024-12-31 23:00"]).to_csv(
        os.path.join(ART, "estacionariedad.csv"), index=False)

    filas_metricas, filas_semillas, filas_sens, filas_abl, filas_base, todas_bit = [], [], [], [], [], []
    for fold in folds:
        p = data.particion(datos, fold)
        cols = data.FEATURES
        norm = data.Normalizador(cols).fit(p["train"])
        tr, va, te = (norm.transform(p[k]) for k in ("train", "val", "test"))
        eda.analisis_variables(p["train"], cols, 24).to_csv(
            os.path.join(ART, f"feature_analysis_fold_{fold}.csv"), index=False)

        # Lineas base triviales, con costos, en validacion y prueba
        for nombre, pol in [("aleatorio", ev.politica_aleatoria(42)), ("siempre_largo", lambda o: 1),
                            ("siempre_plano", lambda o: 0)]:
            for conj, df in (("val", va), ("test", te)):
                env = BTCEnv(df, cols, **env_kw(REJILLA[0], episodio=None))
                bit, _ = ev.correr(env, pol)
                filas_base.append(dict(fold=fold, conjunto=conj, linea_base=nombre,
                                       **ev.metricas(ev.curva_capital(bit, df.index), bit)))

        # Busqueda de hiperparametros en validacion (semilla 42)
        curvas, val_rows, modelos = {}, [], {}
        for i, cfg in enumerate(REJILLA):
            m, curva = entrenar("PPO", tr, cols, 42, pasos, env_kw(cfg), gamma=cfg["gamma"])
            curvas[f"PPO cfg{i} H={cfg['horizonte']} {cfg['modo_reward']} g={cfg['gamma']}"] = curva
            _, _, _, met = evaluar(m, va, cols, cfg)
            val_rows.append(dict(cfg=i, algoritmo="PPO", semilla=42, **cfg, **met))
            modelos[i] = m
        val = pd.DataFrame(val_rows)
        mejor = int(val.sort_values(["sharpe", "retorno_total"], ascending=False).iloc[0]["cfg"])
        cfg = REJILLA[mejor]

        # Semillas adicionales y segundo algoritmo con la configuracion elegida
        candidatos = {("PPO", 42): modelos[mejor]}
        for algo in ("PPO", "DQN"):
            for s in SEMILLAS:
                if (algo, s) in candidatos:
                    continue
                m, curva = entrenar(algo, tr, cols, s, pasos, env_kw(cfg), gamma=cfg["gamma"])
                curvas[f"{algo} seed {s} (cfg{mejor})"] = curva
                candidatos[(algo, s)] = m
        for (algo, s), m in candidatos.items():
            _, _, _, met_v = evaluar(m, va, cols, cfg)
            if not (algo == "PPO" and s == 42):
                val_rows.append(dict(cfg=mejor, algoritmo=algo, semilla=s, **cfg, **met_v))
            # El conjunto de prueba se toca una vez por modelo, con la configuracion ya congelada
            _, _, _, met_t = evaluar(m, te, cols, cfg)
            filas_semillas.append(dict(fold=fold, algoritmo=algo, semilla=s, sharpe_val=met_v["sharpe"],
                                       **{f"test_{k}": v for k, v in met_t.items()}))
        pd.DataFrame(val_rows).to_csv(os.path.join(ART, f"validation_results_fold_{fold}.csv"), index=False)
        plots.curva_entrenamiento(curvas, os.path.join(ART, f"training_curves_fold_{fold}.png"), fold)

        # Modelo oficial del fold: la semilla PPO con mejor Sharpe en validacion
        sem = pd.DataFrame([r for r in filas_semillas if r["fold"] == fold and r["algoritmo"] == "PPO"])
        s_of = int(sem.sort_values("sharpe_val", ascending=False).iloc[0]["semilla"])
        oficial = candidatos[("PPO", s_of)]
        bit, dec, eq, met = evaluar(oficial, te, cols, cfg)
        bit.insert(0, "fold", fold)
        bit.to_csv(os.path.join(ART, f"trade_log_fold_{fold}.csv"), index=False)
        dec.to_csv(os.path.join(ART, f"decisiones_fold_{fold}.csv"), index=False)
        todas_bit.append(bit.assign(semilla=s_of))
        btc = ev.btc_buy_hold(te)
        spx = ev.sp500_buy_hold(sp, te.index[0].normalize(), te.index[-1])
        plots.curvas_capital({"Agente PPO": eq, "Comprar y mantener BTC": btc, "S&P 500": spx},
                             os.path.join(ART, f"equity_curves_test_fold_{fold}.png"),
                             f"Fold {fold}, prueba {te.index[0].date()} a {te.index[-1].date()} (con costos de 0.125% por lado)")
        m_btc, m_spx = ev.metricas(btc), ev.metricas(spx, periodos_anio=252)
        r_ag = eq.resample("1D").last().pct_change()
        r_btc = btc.resample("1D").last().pct_change()
        filas_metricas += [
            dict(fold=fold, estrategia="agente", config=mejor, semilla=s_of, **met,
                 alfa_vs_btc=met["retorno_total"] - m_btc["retorno_total"],
                 alfa_vs_sp500=met["retorno_total"] - m_spx["retorno_total"],
                 corr_diaria_con_btc=r_ag.corr(r_btc)),
            dict(fold=fold, estrategia="btc_buy_hold", **m_btc),
            dict(fold=fold, estrategia="sp500_buy_hold", **m_spx)]

        # Sensibilidad: resolucion intrabarra y retraso de una barra en los datos
        for nombre, extra in [("base (stop primero)", {}), ("take-profit primero", dict(stop_primero=False)),
                              ("datos con 1 barra de retraso", dict(retraso=1))]:
            b2, _, _, m2 = evaluar(oficial, te, cols, cfg, **extra)
            filas_sens.append(dict(fold=fold, escenario=nombre, **ev.resumen_trades(b2), **m2))

        # Ablacion por familia de variables (en validacion, para no reutilizar la prueba)
        for fam, quitar in [("todas", [])] + [(f"sin {k}", v) for k, v in data.FAMILIAS.items()]:
            cols_a = [c for c in cols if c not in quitar]
            m, _ = entrenar("PPO", tr, cols_a, 42, pasos, env_kw(cfg), gamma=cfg["gamma"])
            _, _, _, ma = evaluar(m, va, cols_a, cfg)
            filas_abl.append(dict(fold=fold, variante=fam, **ma))
        print(f"fold {fold} listo: cfg {mejor}, semilla {s_of}, Sharpe test {met['sharpe']:.2f} "
              f"({(time.time() - t_ini) / 60:.1f} min)", flush=True)

    # Agregados entre folds
    mt = pd.DataFrame(filas_metricas)
    agg = mt.groupby("estrategia").agg(["mean", "std"]).drop(columns=["fold", "config", "semilla"], errors="ignore")
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    mt.to_csv(os.path.join(ART, "metrics_summary.csv"), index=False)
    agg.to_csv(os.path.join(ART, "metrics_summary_agregado.csv"))
    pd.DataFrame(filas_semillas).to_csv(os.path.join(ART, "semillas_test.csv"), index=False)
    pd.DataFrame(filas_sens).to_csv(os.path.join(ART, "sensibilidad.csv"), index=False)
    pd.DataFrame(filas_abl).to_csv(os.path.join(ART, "ablacion.csv"), index=False)
    pd.DataFrame(filas_base).to_csv(os.path.join(ART, "lineas_base.csv"), index=False)

    # Analisis de operaciones: peores 3 y periodo de mayor drawdown
    bit = pd.concat(todas_bit, ignore_index=True)
    peores = bit.nsmallest(3, "pnl_neto")
    estado = datos[data.FEATURES].loc[peores["entrada_ts"]].reset_index(drop=True)
    pd.concat([peores.reset_index(drop=True), estado.add_prefix("estado_")], axis=1).to_csv(
        os.path.join(ART, "peores_operaciones.csv"), index=False)
    peor_fold = int(mt[mt.estrategia == "agente"].sort_values("max_drawdown").iloc[0]["fold"])
    b = bit[bit.fold == peor_fold]
    te = data.particion(datos, peor_fold)["test"]
    eq = ev.curva_capital(b, te.index)
    fondo = (eq / eq.cummax() - 1).idxmin()
    pico = eq.loc[:fondo].idxmax()
    ini, fin = pico - pd.Timedelta(days=5), fondo + pd.Timedelta(days=5)
    plots.periodo_drawdown(te["close"].loc[ini:fin], eq.loc[ini:fin],
                           b[(b.entrada_ts >= ini) & (b.entrada_ts <= fin)],
                           os.path.join(ART, "mayor_drawdown.png"),
                           f"Mayor drawdown del agente: fold {peor_fold}, {pico.date()} a {fondo.date()}")

    meta["minutos"] = round((time.time() - t_ini) / 60, 1)
    with open(os.path.join(ART, "run_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print("Listo en", meta["minutos"], "min")


if __name__ == "__main__":
    main()
