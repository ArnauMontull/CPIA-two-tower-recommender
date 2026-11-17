"""Recomendación para un usuario nuevo: top-k por producto escalar (sin lo ya valorado) y reordenación MMR."""
from __future__ import annotations

import numpy as np
import torch

from .modelo import Bolsa, DosTorres


def recomendar_usuario(modelo: DosTorres, V: torch.Tensor, A: tuple, B: tuple, excluidos: np.ndarray, k: int = 20):
    """(items [k], scores [k]) para un usuario nuevo con bolsas A y B = (indices, pesos). Sin lo ya valorado (C14)."""
    bolsa = lambda ix, pw: Bolsa(np.array([0, len(ix)]), ix, pw)
    with torch.no_grad():
        u = modelo.vectores_usuarios(V, torch.tensor([0]), bolsa(*A), bolsa(*B))[0]
        s = (V @ u).numpy().copy()
    s[np.asarray(excluidos, dtype=np.int64)] = -np.inf
    top = np.lexsort((np.arange(len(s)), -s))[:k]
    return top, s[top]


def mmr(V: torch.Tensor, candidatos: np.ndarray, scores: np.ndarray, k: int = 20, lam: float = 0.7) -> np.ndarray:
    """Reordenación MMR (Carbonell y Goldstein): en cada paso, el candidato con mayor lam * score - (1 - lam) * (máxima
    similitud coseno con lo ya elegido). V tiene filas de norma 1, así que la similitud es el producto escalar.
    La relevancia se normaliza a [0, 1] entre los candidatos: los scores del modelo ocupan un rango estrecho (~0,1) y
    las similitudes uno ancho, así que sin normalizar `lam` no equilibraría nada (la diversidad aplastaría al score).
    lam = 1: el ranking original; lam menor: más diversidad. Devuelve k índices de película."""
    C = V[torch.as_tensor(candidatos, dtype=torch.int64)].numpy()
    rel = (scores - scores.min()) / max(float(scores.max() - scores.min()), 1e-12)
    sim = C @ C.T
    elegidos = [0]                                           # el primero es siempre el mejor
    while len(elegidos) < min(k, len(candidatos)):
        resto = np.setdiff1d(np.arange(len(candidatos)), elegidos)
        valor = lam * rel[resto] - (1 - lam) * sim[np.ix_(resto, elegidos)].max(axis=1)
        elegidos.append(int(resto[np.argmax(valor)]))
    return np.asarray(candidatos)[elegidos]
