# hsv_ranges.py
import numpy as np

# HSV ranges (H:0-180, S:0-255, V:0-255)
LOWER = {
    'red':    (0, 110, 90),
    'red2':   (172, 110, 90),  # second red range top
    'green':  (35, 80, 80),
    'blue':   (95, 100, 80),
    'yellow': (22, 110, 110),
    'orange': (8, 130, 130),
    # 'purple': (125, 80, 80), # Удалено
    'black':  (0, 0, 0),
}

UPPER = {
    'red':    (8, 255, 255),
    'red2':   (180, 255, 255),
    'green':  (80, 255, 255),
    'blue':   (125, 255, 255),
    'yellow': (34, 255, 255),
    'orange': (20, 255, 255),
    # 'purple': (155, 255, 255), # Удалено
    'black':  (180, 255, 70),
}

# Colors for drawing (BGR)
DRAW_COLORS = {
    'red':    (0, 0, 255),
    'green':  (0, 255, 0),
    'blue':   (255, 0, 0),
    'yellow': (0, 255, 255),
    'orange': (0, 165, 255),
    # 'purple': (255, 0, 255), # Удалено
    'black':  (80, 80, 80),
}