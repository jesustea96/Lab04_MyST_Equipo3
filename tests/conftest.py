import os
import sys

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)
# 'code' tambien es un modulo de la biblioteca estandar; pytest lo importa antes que nosotros
if "code" in sys.modules and not getattr(sys.modules["code"], "__file__", "").startswith(RAIZ):
    del sys.modules["code"]
os.chdir(RAIZ)
