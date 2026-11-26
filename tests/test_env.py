"""Pruebas obligatorias (seccion 3.11): check_env, causalidad, operacion de referencia e identidad contable."""
import warnings

import numpy as np
import pandas as pd
import pytest
from gymnasium.utils.env_checker import check_env as gym_check_env
from stable_baselines3.common.env_checker import check_env

from code import data, evaluate as ev
from code.env import BTCEnv
from code.reward import retorno_neto


@pytest.fixture(scope="module")
def crudo():
    return data.cargar()


@pytest.fixture(scope="module")
def datos(crudo):
    return data.preparar(crudo)


def test_check_env_sin_advertencias(datos):
    df = data.Normalizador(data.FEATURES).fit(datos.iloc[:5000]).transform(datos.iloc[:5000])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # cualquier advertencia hace fallar la prueba
        check_env(BTCEnv(df, data.FEATURES), warn=True)
        gym_check_env(BTCEnv(df, data.FEATURES), skip_render_check=True)


def test_causalidad(crudo):
    """La observacion en i calculada solo con datos hasta i es igual a la calculada con la serie completa."""
    completa = data.features(crudo)
    for i in (500, 20_000, 60_000):
        parcial = data.features(crudo.iloc[: i + 1])
        a, b = parcial.iloc[i][data.FEATURES].to_numpy(float), completa.iloc[i][data.FEATURES].to_numpy(float)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12, equal_nan=True), f"look-ahead en el indice {i}"


def _mercado_sintetico():
    """6 barras con precios redondos para calcular la operacion a mano."""
    idx = pd.date_range("2024-01-01", periods=6, freq="h")
    df = pd.DataFrame(dict(open=[100, 100, 101, 102, 103, 104.0], high=[101, 101.5, 102, 103, 106, 105.0],
                           low=[99, 99.5, 100.5, 101.5, 102.5, 103.5], close=[100, 101, 102, 103, 104, 104.5]),
                      index=idx)
    df["f"] = 0.0
    df["sigma_ewma_raw"] = 0.01
    return df


def _operar(lado, k_stop, k_tp):
    env = BTCEnv(_mercado_sintetico(), ["f"], horizonte=4, k_stop=k_stop, k_tp=k_tp, episodio=None)
    env.reset(seed=0)
    env.step(1 if lado == 1 else 2)
    return env.bitacora[0]


def test_operacion_de_referencia():
    """Calculado a mano. Entrada al open de la barra 0 = 100; sigma*sqrt(H) = 0.01*sqrt(4) = 0.02.
    a) Largo, stop 96, TP 106: ninguna barra 0-3 los toca -> sale por horizonte al cierre de la barra 3 (103), 4 barras.
    b) Largo, TP = 100*(1+1.0*0.02) = 102: la barra 2 (high 102) lo toca -> take-profit a 102, 3 barras.
    c) Corto, stop = 100*(1+0.5*0.02) = 101: la barra 0 (high 101) lo toca -> stop a 101, 1 barra."""
    a = _operar(1, 2.0, 3.0)
    assert (a["razon"], a["duracion"], a["precio_salida"]) == ("horizonte", 4, 103.0)
    assert a["pnl_neto"] == pytest.approx(100_000 * (1.03 * (1 - 0.00125) ** 2 - 1))

    b = _operar(1, 2.0, 1.0)
    assert (b["razon"], b["duracion"]) == ("take_profit", 3) and b["precio_salida"] == pytest.approx(102.0)
    assert b["pnl_neto"] == pytest.approx(100_000 * (1.02 * (1 - 0.00125) ** 2 - 1))

    c = _operar(-1, 0.5, 3.0)
    assert (c["razon"], c["duracion"]) == ("stop", 1) and c["precio_salida"] == pytest.approx(101.0)
    assert c["pnl_neto"] == pytest.approx(100_000 * retorno_neto(-1, 100.0, 101.0))


def test_identidad_contable(datos):
    df = data.Normalizador(data.FEATURES).fit(datos.iloc[:3000]).transform(datos.iloc[:3000])
    env = BTCEnv(df, data.FEATURES, episodio=None)
    bit, _ = ev.correr(env, ev.politica_aleatoria(7))
    assert len(bit) > 10
    assert env.capital == pytest.approx(env.capital0 + bit["pnl_neto"].sum(), rel=1e-12)
    assert np.allclose(bit["pnl_bruto"] - bit["costos"], bit["pnl_neto"])
