# Lab04_MyST_Equipo3: agente de RL para BTCUSDT horario

Microestructura y Sistemas de Trading, ITESO, otoño 2026.

**Integrantes:** Adrián Ballesteros, Erik del Castillo, Jesús Emmanuel Flores Cortés

**Nivel de alcance:** [COMPLETAR con el nivel acordado en clase]

## Qué hace

Formula la operación de BTCUSDT como un problema de decisión secuencial semi-Markov (marco de Powell). El agente decide solo cuando está plano: no operar durante una hora, abrir largo o abrir corto. Un gestor de posiciones cierra cada operación en el primero de tres eventos: horizonte de H horas, stop-loss o take-profit. El stop y el take-profit se escalan con la volatilidad EWMA. La contribución es terminal (se paga al cierre) e incluye 0.125% de costo por lado. Se entrena con Stable-Baselines3 (PPO y DQN como segundo algoritmo) en cuatro folds walk-forward con validación y prueba separadas.

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

Las versiones de Stable-Baselines3, Gymnasium y PyTorch están fijadas en `requirements.txt` y se registran en `artifacts/run_metadata.json` en cada corrida.

## Reproducción

```bash
python main.py            # pipeline completo, semillas 42, 43 y 44 (~1.5 h en CPU)
python main.py --rapido   # prueba de humo: 3,000 pasos, solo fold 1 (~2 min)
pytest -q tests           # check_env, causalidad, operación de referencia e identidad contable
```

`data/Binance_BTCUSDT_1h.csv` es el archivo publicado en Canvas. `data/sp500.csv` es el S&P 500 diario (^GSPC, Yahoo Finance), descargado una vez y congelado; solo se usa como benchmark.

## Estructura

```
main.py            punto de entrada único
code/data.py       carga, features causales (shift de una barra), folds y normalizador por fold
code/env.py        entorno Gymnasium semi-Markov y gestor de posiciones
code/reward.py     funciones de contribución (riesgo-ajustada y P&L)
code/train.py      entrenamiento SB3 por fold, configuración y semilla
code/evaluate.py   corrida de políticas, benchmarks y métricas
code/eda.py        EDA del entrenamiento y análisis de variables por fold
code/plots.py      gráficas
artifacts/         artefactos exigidos (sección 3.8) y extras (sensibilidad, ablación, semillas, líneas base)
tests/             pruebas con pytest
docs/              reporte.pdf, walkthrough.pdf, presentacion.pdf
```

## Uso de IA

Se usó Claude Code (Anthropic) como asistente de programación para escribir la primera versión de los módulos de `code/`, las pruebas y la estructura del reporte. El diseño del entorno, la formulación, la revisión del código y la interpretación de resultados son responsabilidad del equipo. Cada integrante debe poder explicar cualquier línea.
