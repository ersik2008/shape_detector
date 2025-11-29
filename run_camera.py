# run_camera.py — УНИВЕРСАЛЬНЫЙ ЗАПУСК (работает с любой камерой)
import cv2
from detector import OfflineShapeDetector

def main():
    camera_index = 1
    backend = cv2.CAP_DSHOW

    cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        print("Внешняя камера не найдена. Пробуем встроенную...")
        camera_index = 0
        cap = cv2.VideoCapture(camera_index, backend)

    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("КРИТИЧЕСКАЯ ОШИБКА: ни одна камера не работает!")
        return

    # Настройки
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    ret, frame = cap.read()
    if not ret:
        print("Кадр не пришёл — перезагрузи камеру.")
        cap.release()
        return

    print(f"КАМЕРА {camera_index} УСПЕШНО ЗАПУЩЕНА!")
    print("Робот видит: square, circle, triangle, cylinder, cube, pyramid")
    print("Нажми ESC для выхода\n")

    detector = OfflineShapeDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Потеря кадра — переподключаюсь...")
            cap.release()
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            continue

        frame, detections = detector.detect(frame)

        # Вывод для Flutter — только нужные фигуры
        valid = [d for d in detections if d['shape'] in ('square','circle','triangle','cylinder','cube','pyramid')]

        if valid:
            print("[DETECTIONS]", end=" ")
            for d in valid[:3]:
                print(f"{d['color']}_{d['shape']}:{d['center'][0]},{d['center'][1]}", end=" ")
            print()

        cv2.imshow("Robot Vision — ЧЕМПИОН 2025", frame)

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()