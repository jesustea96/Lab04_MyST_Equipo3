"""Entorno Gymnasium semi-Markov para BTCUSDT horario.

Formulacion (Powell):
- Estado fisico R_t: posicion (siempre 0 en un punto de decision), precio de entrada, tiempo en posicion, capital.
- Estado de informacion I_t: features de mercado hasta el cierre de t-1.
- Estado de creencia B_t: volatilidad EWMA y pendiente de tendencia con su estadistico t.
- Decision x_t en {0: no operar una hora, 1: largo, 2: corto}. No hay acciones invalidas porque el agente
  solo decide estando plano; el gestor de posiciones cierra por horizonte, stop-loss o take-profit.
- Informacion exogena W: las barras OHLC que llegan mientras la posicion esta abierta.
- Transicion: entra al open de t; revisa cada barra; sale en el primer evento (stop, TP, horizonte).
- Contribucion: terminal, al cierre (reward.contribucion). Esperar paga 0.
- Objetivo: maximizar la suma esperada de contribuciones.
Cada step del entorno es una decision; dura 1 barra (esperar) o hasta H barras (operar).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from code.reward import COSTO, contribucion, retorno_neto


class BTCEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, datos: pd.DataFrame, cols: list[str], horizonte: int = 24, k_stop: float = 2.0,
                 k_tp: float = 3.0, episodio: int | None = 720, modo_reward: str = "riesgo",
                 stop_primero: bool = True, costo: float = COSTO, capital: float = 100_000.0,
                 retraso: int = 0):
        super().__init__()
        self.cols, self.H, self.k_stop, self.k_tp = cols, horizonte, k_stop, k_tp
        self.episodio, self.modo, self.stop_primero, self.costo = episodio, modo_reward, stop_primero, costo
        self.capital0 = capital
        o, h, l, c = (datos[k].to_numpy(float) for k in ("open", "high", "low", "close"))
        self.o, self.h, self.l, self.c = o, h, l, c
        x = datos[cols].to_numpy(np.float32)
        if retraso:  # la fuente de datos llega con 'retraso' barras de atraso
            x = np.vstack([np.repeat(x[:1], retraso, 0), x[:-retraso]])
        self.x = x
        self.sigma = datos["sigma_ewma_raw"].to_numpy(float) if "sigma_ewma_raw" in datos else np.full(len(o), 0.01)
        self.idx = datos.index
        self.n = len(o)
        self.action_space = spaces.Discrete(3)
        # Las features vienen normalizadas y recortadas a +-5 (data.Normalizador); la ultima es la fraccion del episodio
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(len(cols) + 1,), dtype=np.float32)

    # ---------- utilidades ----------
    def _obs(self) -> np.ndarray:
        frac = (self.t - self.t0) / max(self.fin - self.t0, 1)  # tiempo transcurrido del episodio
        return np.clip(np.append(self.x[self.t], np.float32(frac)), -5.0, 5.0).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        largo = self.episodio or self.n - 1
        max_ini = max(self.n - 1 - largo, 0)
        self.t0 = int(self.np_random.integers(0, max_ini + 1)) if self.episodio else 0
        self.t = self.t0
        self.fin = min(self.t0 + largo, self.n - 1)
        self.capital = self.capital0
        self.bitacora: list[dict] = []
        return self._obs(), {}

    def simular_operacion(self, t: int, lado: int) -> dict:
        """Gestor de posiciones: entra al open de t y sale en el primer evento. Devuelve el detalle de la operacion."""
        entrada = self.o[t]
        dist = self.sigma[t] * np.sqrt(self.H)
        if lado == 1:
            stop, tp = entrada * (1 - self.k_stop * dist), entrada * (1 + self.k_tp * dist)
        else:
            stop, tp = entrada * (1 + self.k_stop * dist), entrada * (1 - self.k_tp * dist)
        ultimo = min(t + self.H - 1, self.n - 1)
        ambigua = False
        for j in range(t, ultimo + 1):
            toca_stop = self.l[j] <= stop if lado == 1 else self.h[j] >= stop
            toca_tp = self.h[j] >= tp if lado == 1 else self.l[j] <= tp
            if toca_stop and toca_tp:
                ambigua = True
                razon = "stop" if self.stop_primero else "take_profit"
            elif toca_stop:
                razon = "stop"
            elif toca_tp:
                razon = "take_profit"
            else:
                continue
            nivel = stop if razon == "stop" else tp
            # si la barra abre mas alla del nivel (gap), se llena al open, que es peor para el stop
            if j > t and ((lado == 1 and razon == "stop" and self.o[j] < nivel) or
                          (lado == -1 and razon == "stop" and self.o[j] > nivel)):
                nivel = self.o[j]
            return dict(t_sal=j, salida=nivel, razon=razon, ambigua=ambigua, entrada=entrada, lado=lado)
        return dict(t_sal=ultimo, salida=self.c[ultimo], razon="horizonte", ambigua=False, entrada=entrada, lado=lado)

    def step(self, accion):
        accion = int(accion)
        if accion == 0:
            r, dur = 0.0, 1
        else:
            lado = 1 if accion == 1 else -1
            op = self.simular_operacion(self.t, lado)
            r_neto = retorno_neto(lado, op["entrada"], op["salida"], self.costo)
            dur = op["t_sal"] - self.t + 1
            pnl = self.capital * r_neto
            bruto = self.capital * lado * (op["salida"] / op["entrada"] - 1)
            self.bitacora.append(dict(entrada_ts=self.idx[self.t], salida_ts=self.idx[op["t_sal"]], lado=lado,
                                      razon=op["razon"], ambigua=op["ambigua"], precio_entrada=op["entrada"],
                                      precio_salida=op["salida"], tamano=self.capital / op["entrada"],
                                      pnl_bruto=bruto, costos=bruto - pnl, pnl_neto=pnl, duracion=dur,
                                      capital_antes=self.capital))
            self.capital += pnl
            r = contribucion(r_neto, self.sigma[self.t], self.H, self.modo)
        self.t = min(self.t + dur, self.fin)
        terminated = bool(self.capital <= 0.5 * self.capital0)  # ruina
        truncated = bool(self.t >= self.fin) and not terminated  # fin de la ventana de datos
        return self._obs(), float(r), terminated, truncated, {"duracion": dur, "capital": self.capital}
