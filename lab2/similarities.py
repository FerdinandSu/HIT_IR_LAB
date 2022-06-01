from __future__ import division
import numpy as np


def measure_inner_product(v1, v2):
    v2 = v2.T
    return np.dot(v1, v2)


def measure_cosine(v1, v2):
    q = v2.T
    q_norm = np.linalg.norm(q)
    norm = np.linalg.norm(v1)*q_norm
    return np.dot(v1, q)/(norm) if norm != 0 else 0


def measure_jaccard(v1, v2):
    v2_T = v2.T
    dot_prod = np.dot(v1, v2_T)
    v2_norm2 = np.dot(v2, v2_T)
    doc_norm2 = np.dot(v1, v1.T)
    division = doc_norm2+v2_norm2-dot_prod
    return dot_prod/(division) if division != 0 else 0
