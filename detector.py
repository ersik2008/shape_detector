# detector.py — УМНАЯ СИСТЕМА ОПОЗНАНИЯ 2025 (чёрные фигуры — с первого раза!)
import cv2
import numpy as np

class OfflineShapeDetector:
    def __init__(self):
        self.min_area = 1100
        self.max_area = 400000

        # Твои проверенные HSV-диапазоны
        self.color_ranges = {
            "red":    [([0, 110, 90], [8, 255, 255]), ([172, 110, 90], [180, 255, 255])],
            "green":  [([35, 80, 80],  [80, 255, 255])],
            "blue":   [([95, 100, 80], [125, 255, 255])],
            "yellow": [([22, 110, 110],[34, 255, 255])],
            "orange": [([8, 130, 130], [20, 255, 255])],
            "black":  [([0, 0, 0],     [180, 255, 80])],
        }

        self.draw_colors = {
            "red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0),
            "yellow": (0, 255, 255), "orange": (0, 165, 255), "black": (220, 220, 220)
        }

    def get_mask(self, hsv, color_name):
        mask = np.zeros(hsv.shape[:2], np.uint8)
        for low, high in self.color_ranges[color_name]:
            mask |= cv2.inRange(hsv, np.array(low), np.array(high))

        if color_name != "black":
            mask[hsv[:,:,2] < 60] = 0
            mask[hsv[:,:,1] < 65] = 0
        else:
            # УМНАЯ ФИЛЬТРАЦИЯ ЧЁРНОГО — только настоящие чёрные фигуры
            mask &= (hsv[:,:,1] < 70)      # Низкая насыщенность
            mask &= (hsv[:,:,2] < 110)     # Тёмные
            mask &= (hsv[:,:,2] > 8)       # Не чёрный фон

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=6)
        return mask

    def get_features(self, cnt):
        area = cv2.contourArea(cnt)
        if not (self.min_area < area < self.max_area):
            return None

        peri = cv2.arcLength(cnt, True)
        if peri < 50:
            return None

        circularity = 4 * np.pi * area / (peri ** 2)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0

        approx = cv2.approxPolyDP(cnt, 0.022 * peri, True)
        vertices = len(approx)

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / min(w, h)

        ellipse_ratio = 1.0
        if len(cnt) >= 5:
            try:
                ellipse = cv2.fitEllipse(cnt)
                ma, mi = ellipse[1]
                ellipse_ratio = max(ma, mi) / min(ma, mi)
            except:
                pass

        return {
            "area": area,
            "circularity": circularity,
            "solidity": solidity,
            "ellipse_ratio": ellipse_ratio,
            "vertices": vertices,
            "aspect_ratio": aspect
        }

    def classify_smart(self, f):
        if not f:
            return None

        c = f["circularity"]
        e = f["ellipse_ratio"]
        s = f["solidity"]
        v = f["vertices"]
        a = f["aspect_ratio"]

        # 1. КРУГ — самый надёжный
        if c > 0.89 and e < 1.9 and s > 0.93:
            return "circle"

        # 2. ЦИЛИНДР — вытянутый эллипс (главный признак!)
        if e > 2.3 and 0.45 < c < 0.83 and s > 0.83:
            return "cylinder"

        # 3. ТРЕУГОЛЬНИК
        if v == 3 and s > 0.80:
            return "triangle"

        # 4. ПИРАМИДА — 4 вершины + вытянутость
        if v == 4 and a > 1.65 and c < 0.79:
            return "pyramid"

        # 5. КВАДРАТ — 4 вершины + почти квадрат
        if v == 4 and a < 1.55 and c > 0.71 and s > 0.89:
            return "square"

        # 6. КУБ — 5–11 вершин (перспектива!) + высокая выпуклость
        if 5 <= v <= 11 and s > 0.88 and a < 2.5:
            return "cube"

        return None

    def detect(self, frame):
        # Адаптивная предобработка
        blurred = cv2.GaussianBlur(frame, (9, 9), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Умный CLAHE — спасает от переосвещения и теней
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        v_channel = clahe.apply(hsv[:,:,2])
        hsv = np.dstack((hsv[:,:,0], hsv[:,:,1], v_channel))

        detections = []

        # Сначала цветные — они приоритетнее
        for color_name in ["red", "green", "blue", "yellow", "orange"]:
            mask = self.get_mask(hsv, color_name)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                features = self.get_features(cnt)
                shape = self.classify_smart(features)
                if not shape:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue

                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                x, y, w, h = cv2.boundingRect(cnt)

                detections.append({
                    "color": color_name,
                    "shape": shape,
                    "center": (cx, cy),
                    "bbox": (x, y, w, h),
                    "area": features["area"]
                })

                box = np.intp(cv2.boxPoints(cv2.minAreaRect(cnt)))
                cv2.drawContours(frame, [box], 0, self.draw_colors[color_name], 8)
                cv2.putText(frame, f"{color_name.upper()} {shape.upper()}",
                            (x, y-30), cv2.FONT_HERSHEY_DUPLEX, 1.8,
                            self.draw_colors[color_name], 5)

        # Потом чёрный — отдельно, чтобы не мешал
        mask_black = self.get_mask(hsv, "black")
        contours_black, _ = cv2.findContours(mask_black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours_black:
            features = self.get_features(cnt)
            shape = self.classify_smart(features)
            if not shape:
                continue

            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            x, y, w, h = cv2.boundingRect(cnt)

            detections.append({
                "color": "black",
                "shape": shape,
                "center": (cx, cy),
                "bbox": (x, y, w, h),
                "area": features["area"]
            })

            box = np.intp(cv2.boxPoints(cv2.minAreaRect(cnt)))
            cv2.drawContours(frame, [box], 0, self.draw_colors["black"], 8)
            cv2.putText(frame, f"BLACK {shape.upper()}",
                        (x, y-30), cv2.FONT_HERSHEY_DUPLEX, 1.8,
                        self.draw_colors["black"], 5)

        detections.sort(key=lambda x: x["area"], reverse=True)
        return frame, detections[:10]