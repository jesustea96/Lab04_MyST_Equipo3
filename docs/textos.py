"""Redaccion del reporte, del walkthrough y de la presentacion. Las cifras se leen de artifacts/."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def pct(x):
    return f"{100 * x:.1f}%"


def f2(x):
    return f"{x:.2f}"


def cargar(art):
    L = lambda n: pd.read_csv(os.path.join(art, n))
    d = dict(m=L("metrics_summary.csv"), sem=L("semillas_test.csv"), sens=L("sensibilidad.csv"),
             abl=L("ablacion.csv"), base=L("lineas_base.csv"), peores=L("peores_operaciones.csv"),
             reg=L("regimenes.csv"), est=L("estacionariedad.csv"),
             val={k: L(f"validation_results_fold_{k}.csv") for k in range(1, 5)},
             trades={k: L(f"trade_log_fold_{k}.csv") for k in range(1, 5)},
             feat={k: L(f"feature_analysis_fold_{k}.csv") for k in range(1, 5)},
             meta=json.load(open(os.path.join(art, "run_metadata.json"))))
    ag = d["m"][d["m"].estrategia == "agente"]
    d["ag"] = ag
    d["agg"] = d["m"].drop(columns=["fold", "config", "semilla"]).groupby("estrategia").agg(["mean", "std"])
    d["costos"] = pd.DataFrame([dict(fold=k, operaciones=len(t), pnl_bruto=t.pnl_bruto.sum(), costos=t.costos.sum(),
                                     pnl_neto=t.pnl_neto.sum(), pct_largos=(t.lado == 1).mean())
                                for k, t in d["trades"].items()])
    # figura resumen para la presentacion
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.27
    for i, (e, et) in enumerate([("agente", "Agente PPO"), ("btc_buy_hold", "Comprar y mantener BTC"),
                                 ("sp500_buy_hold", "S&P 500")]):
        s = d["m"][d["m"].estrategia == e]
        ax.bar(s.fold + (i - 1) * w, s.retorno_total * 100, width=w, label=et)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Retorno total en prueba por fold (con costos)", xlabel="fold (año de prueba)", ylabel="retorno (%)",
           xticks=[1, 2, 3, 4], xticklabels=["1 (2023)", "2 (2024)", "3 (2025)", "4 (2026 ene-abr)"])
    ax.legend()
    fig.tight_layout()
    d["fig_resumen"] = os.path.join(art, "resumen_retornos.png")
    fig.savefig(d["fig_resumen"], dpi=120)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    c = d["costos"]
    ax.bar(c.fold - 0.2, c.pnl_bruto / 1000, width=0.4, label="P&L bruto")
    ax.bar(c.fold + 0.2, c.pnl_neto / 1000, width=0.4, label="P&L neto (después de costos)")
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="El agente gana antes de costos en los 4 folds; los costos lo vuelven negativo en 2",
           xlabel="fold", ylabel="miles de USDT (capital inicial 100k)", xticks=[1, 2, 3, 4])
    ax.legend()
    fig.tight_layout()
    d["fig_costos"] = os.path.join(art, "bruto_vs_neto.png")
    fig.savefig(d["fig_costos"], dpi=120)
    plt.close(fig)
    return d


def _v(d, est, col, stat="mean"):
    return d["agg"].loc[est, (col, stat)]


def tabla_metricas(d):
    cols = [("retorno_total", pct), ("retorno_anual", pct), ("volatilidad_anual", pct), ("sharpe", f2),
            ("sortino", f2), ("max_drawdown", pct), ("calmar", f2)]
    filas = []
    for e, et in [("agente", "Agente PPO"), ("btc_buy_hold", "BTC comprar y mantener"), ("sp500_buy_hold", "S&P 500")]:
        f = {"Estrategia": et}
        for c, fm in cols:
            f[c] = f"{fm(_v(d, e, c))} ± {fm(_v(d, e, c, 'std'))}"
        filas.append(f)
    extra = {"Estrategia": "Agente: otras"}
    for c, fm in [("hit_rate", pct), ("duracion_media_h", lambda x: f"{x:.1f} h"), ("rotacion_anual", lambda x: f"{x:.0f}x"),
                  ("alfa_vs_btc", pct), ("alfa_vs_sp500", pct), ("corr_diaria_con_btc", f2)]:
        extra[c] = f"{fm(_v(d, 'agente', c))} ± {fm(_v(d, 'agente', c, 'std'))}"
    return pd.DataFrame(filas), pd.DataFrame([extra])


def reporte(d, art):
    ag, m, c, sens, abl = d["ag"], d["m"], d["costos"], d["sens"], d["abl"]
    peor = ag.sort_values("sharpe").iloc[0]
    t1, t2 = tabla_metricas(d)
    sem = d["sem"].groupby(["fold", "algoritmo"]).test_sharpe.agg(["mean", "std"]).reset_index()
    sem_alg = d["sem"].groupby("algoritmo")[["test_sharpe", "test_retorno_total"]].mean()
    ablp = abl.pivot(index="variante", columns="fold", values="sharpe")
    ablp["promedio"] = ablp.mean(axis=1)
    ablp = ablp.reset_index().sort_values("promedio", ascending=False)
    delay = sens[sens.escenario.str.contains("retraso")].set_index("fold").sharpe - sens[
        sens.escenario.str.contains("base")].set_index("fold").sharpe
    feat = d["feat"][1]
    p = d["peores"]
    est = d["est"]
    vol_mi = feat.iloc[0]
    sel = pd.concat([d["val"][k].iloc[:5].assign(fold=k) for k in range(1, 5)])
    elegida = {k: int(ag[ag.fold == k].config.iloc[0]) for k in range(1, 5)}

    return [
        ("h1", "Lab 04: agente de RL para BTCUSDT horario"),
        ("p", "Equipo 3: Adrián Marcelo Ballesteros, Erik del Castillo y Jesús Emmanuel Flores Cortés. Microestructura y Sistemas de "
              "Trading, ITESO, otoño 2026. Repositorio Lab04_MyST_Equipo3; reproducción con `python main.py`."),
        ("h1", "1. Resumen ejecutivo"),
        ("p", f"Formulamos la operación de BTCUSDT como un problema de decisión secuencial semi-Markov y entrenamos PPO "
              f"(con DQN como segundo algoritmo) en cuatro folds walk-forward. Después de costos de 0.125% por lado, el agente "
              f"obtiene un retorno total de prueba de {pct(_v(d, 'agente', 'retorno_total'))} ± "
              f"{pct(_v(d, 'agente', 'retorno_total', 'std'))} (media ± desviación estándar entre folds), contra "
              f"{pct(_v(d, 'btc_buy_hold', 'retorno_total'))} de comprar y mantener BTC y "
              f"{pct(_v(d, 'sp500_buy_hold', 'retorno_total'))} del S&P 500. Su Sharpe medio es "
              f"{f2(_v(d, 'agente', 'sharpe'))} ± {f2(_v(d, 'agente', 'sharpe', 'std'))}."),
        ("p", f"El resultado más importante es que el agente sí gana antes de costos en los cuatro folds (P&L bruto de "
              f"{', '.join(f'{x / 1000:+.1f}k' for x in c.pnl_bruto)} USDT sobre 100k), pero los costos "
              f"({', '.join(f'{x / 1000:.1f}k' for x in c.costos)}) se comen ese margen y lo vuelven negativo en los folds 1 y 4. "
              "No encontramos una ventaja explotable después de costos: lo que el agente aprende es, sobre todo, cuándo no "
              "operar, y cuando opera tiene un sesgo largo que explica su correlación positiva con BTC."),
        ("img", (d["fig_costos"], 15)),

        ("h1", "2. Datos y análisis exploratorio"),
        ("p", "Usamos el archivo de Canvas (BTCUSDT horario, 2017-08-17 04:00 a 2026-04-26 23:00, 75,558 registros) sin "
              "datos externos de precio. El S&P 500 (^GSPC diario, Yahoo Finance) se descarga una vez, se congela en "
              "data/sp500.csv y solo se usa como benchmark. El EDA completo está en artifacts/btcusdt_eda.pdf y usa únicamente "
              "datos de entrenamiento (hasta 2024-12)."),
        ("h2", "Pregunta 1. Regímenes del entrenamiento"),
        ("tabla", (d["reg"].assign(retorno_total=d["reg"].retorno_total.map(pct), vol_anual=d["reg"].vol_anual.map(pct),
                                   tendencia_diaria=d["reg"].tendencia_diaria.map(lambda x: f"{100 * x:.2f}%"),
                                   max_drawdown=d["reg"].max_drawdown.map(pct)),)),
        ("p", "Identificamos siete regímenes por rango de fechas. Difieren mucho en volatilidad (de 47% a 130% anual), en "
              "tendencia y en drawdown: el invierno de 2018 y el bajista de 2022 tuvieron caídas de 83% y 76%, mientras que el "
              "toro 2020-2021 multiplicó el precio casi 14 veces. La volatilidad baja con el tiempo, así que los folds de prueba "
              "recientes (2023-2026) son un régimen más tranquilo que buena parte del entrenamiento."),
        ("h2", "Pregunta 2. Estacionariedad, autocorrelación y elección del estado de información"),
        ("tabla", (est.round(4),)),
        ("p", f"El log precio no es estacionario (ADF p = {est.adf_p[0]:.2f}; KPSS rechaza estacionariedad), y el retorno "
              f"horario sí lo es (ADF p < 0.01, KPSS no rechaza). Por eso ninguna variable del estado usa niveles de precio: "
              "todas son retornos, distancias relativas o volatilidades. La autocorrelación de los retornos es casi nula, "
              "pero la del valor absoluto es alta y persistente, con un ciclo de 24 horas. Lo confirma el análisis de variables "
              f"del fold 1: la variable con más información mutua con el retorno de las siguientes 24 h es {vol_mi.variable} "
              f"(IM = {vol_mi.info_mutua:.3f}), pero su correlación con el signo del retorno es apenas {vol_mi.spearman:.3f} y "
              f"con su magnitud es {vol_mi.corr_abs_y:.2f}. Es decir, los datos predicen cuánto se va a mover el precio, no "
              "hacia dónde. Por eso el estado pone el peso en volatilidad (4 variables, más la creencia EWMA) y deja pocas de "
              "momentum. La alternativa que rechazamos fue la plantilla de la S18: una ventana de 30 retornos crudos. Con "
              "autocorrelación casi nula, esa ventana agrega 30 dimensiones de ruido y ninguna medida de volatilidad."),

        ("h1", "3. Formulación como problema de decisión secuencial (Powell)"),
        ("p", "El agente decide solo cuando está plano. Cada decisión dura una hora (no operar) o hasta H horas (operar). "
              "Esa duración variable es lo que hace al problema semi-Markov."),
        ("b", "Estado S_n = (R_n, I_n, B_n) en el punto de decisión n, que ocurre a la hora t_n."),
        ("b", "Estado físico R_n: posición p ∈ {−1, 0, +1} (siempre 0 al decidir), precio de entrada, tiempo en posición y "
              "capital K_n. Durante la posición evolucionan con la transición; el agente no los observa porque al decidir "
              "siempre valen lo mismo, excepto el capital, que no afecta la política porque el tamaño es fijo (100% del capital, "
              "sin apalancamiento)."),
        ("b", "Estado de información I_n: retornos log de 1, 6, 24 y 168 h, distancia a la media de 168 h, RSI(14), "
              "volatilidades de 24 y 168 h, Parkinson de 24 h, razón de volatilidades, z-score del volumen y del número de "
              "operaciones, y la hora del día (seno y coseno). Todas usan datos hasta el cierre de t_n − 1."),
        ("b", "Estado de creencia B_n: volatilidad EWMA (vida media de 24 h), que es la estimación de σ, y pendiente de una "
              "regresión de 168 h con su estadístico t como medida de la incertidumbre sobre la tendencia."),
        ("b", "Decisión x_n ∈ {0: no operar 1 h, 1: abrir largo, 2: abrir corto}. No hay acciones inválidas: el agente nunca "
              "decide con posición abierta, y el cierre lo hace el gestor de posiciones. Rechazamos una acción explícita de "
              "cierre porque obliga a observar la posición abierta y el P&L no realizado, rompe la estructura semi-Markov (el "
              "agente tendría que decidir cada hora) y permite cerrar y reabrir pagando costos dos veces. No la probamos "
              "empíricamente; es una decisión de diseño."),
        ("b", "Información exógena W: las barras OHLC que llegan mientras la posición está abierta."),
        ("b", "Transición S^M: entrada al open de t_n. Se fija stop = entrada·(1 ∓ k_s·σ̂·√H) y take-profit = "
              "entrada·(1 ± k_tp·σ̂·√H), con k_s = 2 y k_tp = 3. Se recorre barra por barra y se sale en el primer evento: stop, "
              "take-profit u horizonte H (cierre de la barra t_n + H − 1). Si el stop y el TP caen en la misma barra, se supone "
              "que el stop se ejecuta primero. Si una barra abre más allá del stop (gap), se llena al open. La siguiente "
              "decisión ocurre en t_{n+1} = t_n + τ_n, donde τ_n es la duración real."),
        ("b", "Contribución C(S_n, x_n, W): terminal, pagada al cierre. Con r = retorno neto de la operación "
              "(1 + lado·(P_salida/P_entrada − 1))·(1 − 0.00125)² − 1, la versión ajustada por riesgo es "
              "C = clip(r / (σ̂·√H), −5, 5) y la de P&L es C = 100·r. No operar paga 0."),
        ("b", "Objetivo: max_π E[Σ_n C(S_n, X^π(S_n), W_{n+1})] sobre la ventana del episodio."),
        ("h2", "Decisiones de diseño y alternativas rechazadas"),
        ("b", "Episodio: ventanas de 720 horas (30 días) con inicio aleatorio dentro del entrenamiento; termina por ruina "
              "(capital ≤ 50%) o se trunca al final de la ventana. Alternativa rechazada: un solo episodio con todo el "
              "entrenamiento, porque da muy pocas actualizaciones y el agente memoriza una sola trayectoria. Lo que perdemos: "
              "la composición del capital a largo plazo y las transiciones entre regímenes dentro de un mismo episodio."),
        ("b", f"Periodo de tenencia: H ∈ {{12, 24}} se eligió en validación (columna horizonte de validation_results). La "
              f"configuración ganadora cambió entre folds ({', '.join(f'fold {k}: cfg {v}' for k, v in elegida.items())}), "
              "lo que ya es una señal de que no hay una configuración robusta."),
        ("b", "Contribución: comparamos la versión ajustada por riesgo con la de P&L en la misma rejilla. Ninguna domina "
              "en los cuatro folds. Elegimos la terminal y no por paso porque, con contribución por paso, el agente recibe "
              "crédito por movimientos que no dependen de su decisión y aprende a abrir para cobrar la recompensa intermedia; "
              "con la terminal, toda la recompensa se asigna a la decisión que abrió la operación (pregunta 3)."),
        ("b", "Descuento: γ = 0. En cada punto de decisión el agente está plano y el mercado es exógeno, así que la acción no "
              "cambia la distribución del siguiente estado y la ecuación de Bellman se reduce a max_x E[C | S, x]. Con γ > 0 el "
              "descuento sería por decisión y no por hora, lo que en un problema semi-Markov favorece operar (24 h cuentan "
              "igual que 1 h de espera). Lo probamos (cfg 4, γ = 0.9) y en validación fue la peor o de las peores en tres de "
              "cuatro folds."),
        ("b", "Replay semi-Markov: cada step del entorno va de la apertura al cierre, así que el buffer de DQN guarda "
              "exactamente (estado en la apertura, acción, recompensa al cierre, estado al cierre); la duración real queda en "
              "info y en la bitácora."),

        ("h1", "4. Entrenamiento y validación"),
        ("p", f"Stable-Baselines3 {d['meta']['stable_baselines3']}, Gymnasium {d['meta']['gymnasium']}, PyTorch "
              f"{d['meta']['torch']} (CPU), {d['meta']['pasos']:,} pasos por modelo, semillas 42, 43 y 44. El entorno pasa "
              "check_env sin advertencias (prueba 1). La normalización usa media y desviación del entrenamiento de cada fold, "
              "se recorta a ±5 y se aplica sin cambios a validación y prueba; no usamos VecNormalize."),
        ("p", "Líneas base con costos (artifacts/lineas_base.csv): el agente aleatorio llega a la ruina (−50%) en los "
              "cuatro folds de prueba y siempre largo también, a pesar de que BTC subió 155% en 2023. Eso confirma que el "
              "simulador no regala dinero: con rotación diaria, 0.25% por operación completa destruye cualquier estrategia "
              "sin señal. Siempre plano da 0%."),
        ("p", "Búsqueda de hiperparámetros: rejilla de 5 configuraciones (H × contribución, más γ = 0.9) con PPO, semilla 42, "
              "evaluada en validación; se elige la de mayor Sharpe. Con la configuración congelada se entrenan dos semillas "
              "más de PPO y tres de DQN. El modelo oficial de cada fold es la semilla PPO con mejor Sharpe en validación, y el "
              "conjunto de prueba se toca una sola vez por modelo."),
        ("tabla", (sem.round(2).rename(columns={"mean": "Sharpe prueba (media)", "std": "desv. entre semillas"}),)),
        ("p", f"La dispersión entre semillas es grande: en varios folds la desviación del Sharpe entre semillas es del mismo "
              f"tamaño que la media. En las curvas de entrenamiento (training_curves_fold_k.png) las configuraciones de PPO "
              f"salen de la zona negativa en las primeras 10 a 15 mil decisiones y después oscilan cerca de cero o un poco "
              f"arriba. Las de DQN terminan con las recompensas de entrenamiento más altas y aun así son las peores en "
              f"prueba: es sobreajuste a las ventanas de entrenamiento, no aprendizaje que generalice."),

        ("h1", "5. Resultados en prueba"),
        ("tabla", (t1,)),
        ("tabla", (t2,)),
        ("h2", "Pregunta 4. Variación entre folds"),
        ("tabla", (ag[["fold", "retorno_total", "sharpe", "max_drawdown", "operaciones", "hit_rate", "corr_diaria_con_btc"]]
                   .assign(retorno_total=lambda x: x.retorno_total.map(pct), max_drawdown=lambda x: x.max_drawdown.map(pct),
                           hit_rate=lambda x: x.hit_rate.map(pct)).round(2),)),
        ("p", f"El fold más débil es el {int(peor.fold)} (prueba 2023, Sharpe {f2(peor.sharpe)}). Ese año BTC subió 155% en "
              f"una tendencia casi continua, y el agente, entrenado hasta 2021 con validación en el bajista de 2022, operó "
              f"{int(peor.operaciones)} veces con horizonte de 24 h. Su P&L bruto fue apenas {c.pnl_bruto[0] / 1000:+.1f}k y pagó "
              f"{c.costos[0] / 1000:.1f}k en costos. Ganó poco cuando estaba largo y perdió en los cortos contra la tendencia "
              "(la peor operación de todo el periodo de prueba es un corto de marzo de 2023). Los folds 2 y 3 ganan apenas "
              "por encima de cero y el 4 pierde: no es una estrategia que gane de forma consistente."),
        ("img", (os.path.join(art, "equity_curves_test_fold_1.png"), 15)),
        ("h2", "Pregunta 5. ¿Supera a los benchmarks después de costos?"),
        ("p", f"No. El alfa medio contra BTC es {pct(_v(d, 'agente', 'alfa_vs_btc'))} y contra el S&P 500 es "
              f"{pct(_v(d, 'agente', 'alfa_vs_sp500'))}. El agente solo le gana a BTC en los folds 3 y 4, cuando BTC cayó, y "
              "porque estuvo fuera del mercado la mayor parte del tiempo. Lo que mide nuestro resultado es, sobre todo, el "
              "costo de transacción: hay una señal débil de corto plazo (el P&L bruto es positivo en los cuatro folds), pero "
              "su tamaño es menor que 0.25% por operación completa."),

        ("h1", "6. Análisis de operaciones"),
        ("h2", "Las tres operaciones con mayor pérdida"),
        ("tabla", (p[["fold", "entrada_ts", "precio_entrada", "salida_ts", "precio_salida", "razon", "lado", "pnl_neto",
                      "estado_ret_24", "estado_vol_ratio", "estado_volz_24", "estado_tend_t"]].round(3),)),
        ("b", f"1) Corto el {p.entrada_ts[0]} a {p.precio_entrada[0]:,.0f}, cerrado por stop a {p.precio_salida[0]:,.0f}: es "
              "el rebote de BTC tras la quiebra de Silicon Valley Bank. Al entrar, el retorno de 24 h era +14% y la volatilidad "
              "de 24 h estaba 70% arriba de la de una semana: el agente leyó el salto como sobrecompra y apostó a una reversión. "
              "Falta en el estado algo que distinga un choque de noticias de un exceso técnico (por ejemplo, la persistencia del "
              "volumen anormal), y un stop de 2σ√H es demasiado amplio cuando σ̂ se dispara."),
        ("b", f"2) Largo el {p.entrada_ts[1]} a {p.precio_entrada[1]:,.0f}, stop a {p.precio_salida[1]:,.0f} en 10 horas: "
              "fue la caída posterior al anuncio de la reserva estratégica de cripto en EE.UU. de marzo de 2025. El estado no "
              "mostraba nada extremo; es el tipo de pérdida que no se puede evitar con estas variables, y que solo un límite "
              "de pérdida por día o un tamaño escalado por volatilidad reduciría."),
        ("b", f"3) Corto el {p.entrada_ts[2]} a {p.precio_entrada[2]:,.0f}, cerrado por horizonte a {p.precio_salida[2]:,.0f}: "
              f"la recuperación tras el desplome de agosto de 2024. La tendencia de 168 h era muy negativa (t = "
              f"{p.estado_tend_t[2]:.1f}) y el agente la siguió justo cuando se agotaba. Una variable de agotamiento de "
              "tendencia o una penalización por operar contra una volatilidad que ya está bajando ayudaría."),
        ("h2", "Periodo de mayor drawdown"),
        ("img", (os.path.join(art, "mayor_drawdown.png"), 15)),
        ("p", "El mayor drawdown ocurre en el fold 1 (2023). La gráfica muestra que no es una sola operación mala sino un goteo: "
              "muchas operaciones de 24 h con resultado bruto cercano a cero, cada una pagando 0.25%. Es el costo, no el "
              "riesgo, lo que hunde la curva."),
        ("h2", "Correlación con BTC"),
        ("p", f"La correlación diaria entre los retornos del agente y los de BTC es {f2(_v(d, 'agente', 'corr_diaria_con_btc'))} "
              f"en promedio (entre {f2(ag.corr_diaria_con_btc.min())} y {f2(ag.corr_diaria_con_btc.max())}). Entre "
              f"{pct(c.pct_largos.min())} y {pct(c.pct_largos.max())} de sus operaciones son largas. Es moderada: el agente no es "
              "una posición larga apalancada, pero sí una posición larga filtrada que pasa la mayor parte del tiempo fuera."),

        ("h1", "7. Sensibilidad y análisis de fallas de producción"),
        ("h2", "Pregunta 6. Resolución intrabarra y retraso de una barra"),
        ("p", "Supuesto intrabarra: si el stop y el take-profit caen en la misma barra, se ejecuta primero el stop. En las "
              "operaciones de prueba de los cuatro folds, la proporción afectada es 0%: con stop a 2σ√H y TP a 3σ√H, la "
              "distancia entre los dos es de 5σ√H, más de lo que se mueve una sola barra horaria. Por eso el resultado es "
              "idéntico bajo el supuesto contrario (artifacts/sensibilidad.csv). Si se acercaran los niveles, esto sí importaría."),
        ("p", f"Retraso de una barra: si la fuente de datos llega una hora tarde, el Sharpe cambia en "
              f"{', '.join(f'{v:+.2f}' for v in delay.values)} (folds 1 a 4). Los cambios son tan grandes como el propio resultado "
              "y de signo distinto según el fold, lo que confirma que no hay una señal robusta: con una ventaja real, el retraso "
              "la degradaría de forma consistente."),
        ("h2", "Ejecución, latencia, capacidad y cambio de régimen"),
        ("b", "Ejecución: el simulador llena al open de la barra, sin spread ni impacto, y llena los stops exactamente en su "
              "nivel salvo gaps. Con 100k USDT BTCUSDT absorbe la orden sin problema, pero a partir de unos millones por orden "
              "el impacto y el deslizamiento en los stops (que se disparan justo cuando la liquidez se va) agregarían costos "
              "que el modelo no tiene."),
        ("b", "Latencia: decidimos con barras horarias, así que 2 segundos de inferencia no cambian el precio de llenado de "
              "forma material. El riesgo real es que los datos lleguen tarde: el experimento de una barra de retraso muestra "
              "cambios de Sharpe de hasta 0.6."),
        ("b", "Capacidad: con 50 fondos operando algo parecido, una señal tan delgada desaparecería. Del otro lado de nuestras "
              "operaciones están market makers y traders de mayor frecuencia que cobran el spread; no hay razón para que nos "
              "cedan una ventaja que ni siquiera cubre los costos."),
        ("b", "Cambio de régimen: el entrenamiento contiene caídas como las de 2018 y 2022, pero el modelo se elige en un año "
              "de validación. Si los próximos 12 meses se parecen a 2018 (volatilidad > 100% y caída de 80%), las variables de "
              "volatilidad normalizadas se saldrían del rango del entrenamiento reciente (se recortan a ±5) y el stop de 2σ√H "
              "se volvería muy amplio. Las variables de volumen se degradarían más, porque su nivel depende de cómo crece el "
              "mercado."),

        ("h1", "8. Pregunta 7. Ablación y segundo algoritmo"),
        ("tabla", (ablp.round(2),)),
        ("p", "La ablación quita una familia de variables, reentrena con la configuración elegida y evalúa en validación. Con "
              "todas las variables, el Sharpe medio de validación es el más alto. Quitar volumen y creencia es lo que más "
              "daña en promedio; quitar calendario o momentum casi no cambia el promedio, pero sí mueve mucho folds "
              "individuales. Con una sola semilla por variante, estas diferencias tienen el mismo orden que la dispersión entre "
              "semillas, así que no las tomamos como concluyentes."),
        ("p", f"DQN fue peor que PPO: Sharpe medio de prueba {f2(sem_alg.loc['DQN', 'test_sharpe'])} contra "
              f"{f2(sem_alg.loc['PPO', 'test_sharpe'])}. En el fold 1 quedaron casi empatados y en los otros tres DQN fue claramente "
              "peor. Nuestra hipótesis, no verificada, es que con una recompensa terminal tan ruidosa el max de DQN "
              "sobreestima el valor de operar (sesgo de maximización)."),

        ("h1", "9. Distancia entre el simulador y la operación real"),
        ("p", "El simulador modela la política y supone que el resto del sistema es gratuito, instantáneo y perfecto: "
              "llenado al open sin spread, stops sin deslizamiento, datos sin retraso, sin caídas del exchange y sin límites de "
              "capital. Con los experimentos de la sección 7, un retraso de una barra cambia el Sharpe hasta en 0.6, el "
              "supuesto intrabarra no cambia nada con estos niveles y los costos de 0.125% por lado ya eliminan toda la "
              "ventaja bruta. Cualquier fricción adicional (spread, slippage) haría el resultado más negativo. Nuestra "
              "conclusión es que el desempeño observado es un artefacto de la muestra, no una ventaja."),
        ("h1", "10. Uso de IA"),
        ("p", "Usamos Claude Code como asistente de programación para la primera versión de los módulos, las pruebas y la "
              "estructura de este reporte. Las decisiones de diseño, la revisión del código y la interpretación son del equipo."),
    ]


def walkthrough(d, art):
    return [
        ("h1", "Recorrido del código: Lab04_MyST_Equipo3"),
        ("p", "Este documento explica cómo está organizado el código, qué hace cada módulo y las decisiones de implementación "
              "que no son evidentes al leerlo."),
        ("h2", "Flujo de main.py"),
        ("b", "1. Carga el csv de Canvas (data.cargar), calcula las features (data.features) y descarta las primeras 168 horas."),
        ("b", "2. Genera el EDA (eda.eda_pdf) solo con datos hasta 2024-12."),
        ("b", "3. Para cada fold: parte en entrenamiento, validación y prueba (data.particion); ajusta el Normalizador con "
              "el entrenamiento; escribe feature_analysis_fold_k.csv; corre las líneas base; entrena la rejilla y elige la "
              "configuración en validación; entrena semillas extra y DQN; evalúa en prueba; corre sensibilidad y ablación."),
        ("b", "4. Agrega métricas entre folds, busca las peores operaciones, grafica el mayor drawdown y guarda "
              "run_metadata.json con las versiones."),
        ("h2", "code/data.py"),
        ("b", "cargar(): el csv viene en orden descendente y con una línea extra de encabezado; se ordena y se quitan duplicados."),
        ("b", "features(): todas las variables se calculan y al final se desplazan una barra (shift(1)). Así, la fila t "
              "contiene información hasta el cierre de t − 1 y la orden se ejecuta al open de t. La hora del día no se "
              "desplaza porque se conoce en t. La prueba de causalidad recalcula las features con datos hasta i y verifica "
              "que coincidan con el cálculo sobre la serie completa."),
        ("b", "sigma_ewma_raw: copia sin normalizar de la volatilidad EWMA. La usa el gestor de posiciones para fijar stop y "
              "TP; el agente ve la versión normalizada."),
        ("b", "Normalizador: z-score con estadísticos del entrenamiento del fold, recortado a ±5. Por eso el "
              "observation_space es Box(−5, 5), lo que también evita la advertencia de check_env."),
        ("h2", "code/env.py"),
        ("b", "BTCEnv.step(): si la acción es 0 avanza una hora con recompensa 0. Si es 1 o 2 llama a simular_operacion(), "
              "que recorre las barras desde t hasta t + H − 1 y devuelve la salida. El entorno salta directamente a la barra "
              "de salida: un step = una decisión, que es lo que hace al problema semi-Markov."),
        ("b", "Stop y TP en la misma barra: se resuelve con stop_primero (True por defecto) y se marca la operación como "
              "ambigua en la bitácora para poder cuantificarlas."),
        ("b", "Gaps: si una barra abre más allá del stop, se llena al open, que es peor que el nivel."),
        ("b", "terminated = capital ≤ 50% (ruina); truncated = fin de la ventana. Nunca se marca terminated por fin de datos."),
        ("b", "retraso: desplaza la matriz de observaciones para simular datos que llegan tarde, sin tocar los precios de llenado."),
        ("h2", "code/reward.py"),
        ("b", "retorno_neto(): aplica 0.125% en la entrada y en la salida de forma multiplicativa. pnl_bruto − costos = "
              "pnl_neto se verifica en la prueba de identidad contable."),
        ("b", "contribucion(): versión 'riesgo' (retorno / σ̂√H, recortado a ±5) o 'pnl' (retorno en %)."),
        ("h2", "code/train.py, evaluate.py, plots.py, eda.py"),
        ("b", "train.entrenar(): fija semillas (set_random_seed, reset y action_space), envuelve el entorno en Monitor para "
              "la curva de aprendizaje y entrena en CPU."),
        ("b", "evaluate.correr(): recorre todo el periodo con episodio=None y la política determinista. curva_capital() "
              "cambia el capital solo al cierre de cada operación, consistente con la contribución terminal."),
        ("b", "Métricas: sobre la curva diaria; anualización con 365 días para BTC y el agente y 252 para el S&P 500. "
              "Rotación = 2 × operaciones por año (entrada y salida de todo el capital)."),
        ("b", "eda.analisis_variables(): el objetivo es el retorno de las siguientes H horas desde el open de t. Usa datos "
              "futuros, pero solo para medir relevancia sobre el entrenamiento y nunca entra al estado."),
        ("h2", "Detalles no evidentes"),
        ("b", "El paquete se llama code/, como pide el enunciado, pero 'code' también es un módulo de la biblioteca estándar. "
              "main.py y tests/conftest.py ponen la raíz del repo al inicio de sys.path y sacan el módulo estándar de "
              "sys.modules si ya estaba cargado."),
        ("b", "torch se instala desde el índice de CPU (primera línea de requirements.txt) para no bajar CUDA."),
        ("b", "Tiempo de corrida: unos 60 minutos en CPU para los cuatro folds, con 60,000 pasos por modelo."),
        ("h2", "Pruebas (pytest -q tests)"),
        ("b", "test_check_env_sin_advertencias: check_env de SB3 y de Gymnasium con las advertencias convertidas en errores."),
        ("b", "test_causalidad: índices 500, 20,000 y 60,000."),
        ("b", "test_operacion_de_referencia: tres operaciones calculadas a mano sobre seis barras sintéticas (salida por "
              "horizonte, por take-profit y por stop) con precio, duración y P&L exactos."),
        ("b", "test_identidad_contable: capital final = capital inicial + suma del P&L neto de la bitácora."),
    ]


def presentacion(d, art):
    c, ag = d["costos"], d["ag"]
    sens = d["sens"]
    delay = sens[sens.escenario.str.contains("retraso")].set_index("fold").sharpe - sens[
        sens.escenario.str.contains("base")].set_index("fold").sharpe
    A = lambda n: os.path.join(art, n)
    return [
        dict(titulo="Lab 04: agente de RL para BTCUSDT", texto=[
            "Equipo 3: Adrián Marcelo Ballesteros, Erik del Castillo, Jesús Flores",
            "Microestructura y Sistemas de Trading, ITESO, otoño 2026"]),
        dict(titulo="Resumen: no hay ventaja después de costos", img=d["fig_resumen"], texto=[
            f"Agente: {pct(_v(d, 'agente', 'retorno_total'))} ± {pct(_v(d, 'agente', 'retorno_total', 'std'))}",
            f"BTC: {pct(_v(d, 'btc_buy_hold', 'retorno_total'))}   S&P 500: {pct(_v(d, 'sp500_buy_hold', 'retorno_total'))}",
            f"Sharpe del agente: {f2(_v(d, 'agente', 'sharpe'))} ± {f2(_v(d, 'agente', 'sharpe', 'std'))}"]),
        dict(titulo="Datos y regímenes", img=A("training_curves_fold_1.png") if False else None, texto=[
            "BTCUSDT horario, 2017-08 a 2026-04 (75,558 barras)",
            "7 regímenes en el entrenamiento: volatilidad de 47% a 130% anual",
            "Caídas de 83% (2018) y 76% (2022); toro 2020-21 de casi 14x"]),
        dict(titulo="Qué predicen los datos: magnitud, no dirección", texto=[
            "Log precio no estacionario (ADF p = 0.93); retornos sí",
            "ACF de retornos ≈ 0; ACF de |retornos| alta, con ciclo de 24 h",
            "Mayor información mutua: volatilidad de 168 h",
            "Correlación con el signo del retorno futuro: 0.02; con su magnitud: 0.33",
            "Por eso el estado pesa volatilidad y rechazamos la ventana de 30 retornos crudos"]),
        dict(titulo="Formulación semi-Markov (Powell)", texto=[
            "Estado: físico (posición, entrada, tiempo, capital) + información (15 features causales) + creencia (σ EWMA, tendencia y su t)",
            "Decisión al estar plano: esperar 1 h, largo o corto",
            "Cierre: horizonte H, stop 2σ√H o TP 3σ√H (stop primero si empatan)",
            "Contribución terminal con 0.125% por lado; γ = 0 porque la acción no cambia el siguiente estado"]),
        dict(titulo="Validación del simulador", texto=[
            "check_env sin advertencias, causalidad, operación de referencia e identidad contable: 4 pruebas pasan",
            "Aleatorio y siempre largo llegan a la ruina (−50%) en los 4 folds",
            "Siempre largo quiebra incluso en 2023, con BTC +155%: los costos mandan"]),
        dict(titulo="Entrenamiento: PPO, 4 folds, 3 semillas", img=A("training_curves_fold_2.png"), texto=[
            "Rejilla: H ∈ {12, 24} × recompensa {riesgo, P&L} + γ = 0.9",
            "Selección por Sharpe en validación; prueba una sola vez",
            "La configuración ganadora cambia entre folds"]),
        dict(titulo="El agente gana antes de costos", img=d["fig_costos"], texto=[
            "P&L bruto positivo en los 4 folds",
            f"Costos: {', '.join(f'{x / 1000:.0f}k' for x in c.costos)} USDT",
            "La señal existe pero es menor que 0.25% por operación"]),
        dict(titulo="Fold más débil: 2023", img=A("equity_curves_test_fold_1.png"), texto=[
            f"Sharpe {f2(ag.sharpe.iloc[0])}, {int(ag.operaciones.iloc[0])} operaciones de 24 h",
            "Cortos contra una tendencia de +155%",
            "El drawdown es un goteo de costos, no una mala operación"]),
        dict(titulo="Peores operaciones", img=A("mayor_drawdown.png"), texto=[
            "Corto en el rebote de SVB (mar 2023): leyó noticia como sobrecompra",
            "Largo en la caída de mar 2025: no había señal en el estado",
            "Corto al final del desplome de ago 2024: siguió una tendencia agotada"]),
        dict(titulo="Sensibilidad", texto=[
            "Intrabarra: 0% de operaciones afectadas (stop y TP separados 5σ√H)",
            f"1 barra de retraso: Sharpe cambia {', '.join(f'{v:+.2f}' for v in delay.values)}",
            "Cambios del tamaño del resultado y de signo variable: no es una señal robusta"]),
        dict(titulo="Ablación y segundo algoritmo", texto=[
            "Todas las familias juntas: mejor Sharpe medio de validación",
            "Quitar volumen o creencia es lo que más daña",
            f"DQN peor que PPO (Sharpe medio {f2(d['sem'].groupby('algoritmo').test_sharpe.mean()['DQN'])})"]),
        dict(titulo="Distancia a producción", texto=[
            "Supone llenado al open, sin spread ni slippage, datos a tiempo",
            "Latencia de 2 s irrelevante a frecuencia horaria; el retraso de datos sí importa",
            "Con 50 fondos haciendo lo mismo, la señal desaparece",
            "Un año tipo 2018 sacaría la volatilidad del rango de entrenamiento"]),
        dict(titulo="Conclusiones", texto=[
            "El agente no supera a BTC ni al S&P 500 después de costos",
            "Hay una señal débil de corto plazo, más chica que el costo de operar",
            "Lo que aprende el agente es cuándo no operar",
            "El resultado observado es un artefacto de la muestra, no una ventaja"]),
    ]
