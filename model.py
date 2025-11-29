# model.py — ПРОТОТИПНЫЙ КЛАССИФИКАТОР (работает под любым ракурсом!)
import numpy as np

SCALES = np.array([1.0, 1.0, 3.0, 5.0, 2.5])

PROTOTYPES = {
    'circle':    np.array([[0.93, 0.96, 1.05, 12.0, 1.05],
                           [0.90, 0.94, 1.10, 10.0, 1.00]]),

    'square':    np.array([[0.75, 0.95, 1.30, 4.0, 1.10],
                           [0.70, 0.93, 1.50, 4.0, 1.15]]),

    'cube':      np.array([[0.68, 0.90, 1.80, 6.0, 1.40],   # куб под углом выглядит как шестиугольник
                           [0.65, 0.88, 2.10, 6.0, 1.60]]),

    'triangle':  np.array([[0.58, 0.88, 1.60, 3.0, 1.20]]),

    'pyramid':   np.array([[0.55, 0.85, 1.90, 4.0, 1.50],   # пирамида даёт 4 вершины + вытянутость
                           [0.52, 0.83, 2.20, 4.0, 1.70]]),

    'cylinder':  np.array([[0.72, 0.92, 2.80, 8.0, 2.10],
                           [0.68, 0.90, 3.50, 7.0, 2.40],
                           [0.65, 0.88, 4.20, 6.0, 2.80]]),
}

THRESH = {
    'circle': 0.55, 'square': 0.70, 'cube': 0.80,
    'triangle': 0.85, 'pyramid': 0.90, 'cylinder': 0.75
}

def predict(features):
    f = np.array([features["circularity"],
                  features["solidity"],
                  features["ellipse_ratio"],
                  float(features["vertices"]),
                  features["aspect_ratio"]])

    best_label = None
    best_dist = float('inf')

    for label, protos in PROTOTYPES.items():
        diffs = (protos - f) / SCALES
        dists = np.linalg.norm(diffs, axis=1)
        min_dist = np.min(dists)
        if min_dist < best_dist:
            best_dist = min_dist
            best_label = label

    if best_label and best_dist <= THRESH[best_label]:
        confidence = max(0.0, 1.0 - best_dist / THRESH[best_label])
        return best_label, confidence
    return None, 0.0