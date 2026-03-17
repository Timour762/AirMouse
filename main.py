import cv2

from config import CAMERA_HEIGHT, CAMERA_INDEX, CAMERA_WIDTH, WINDOW_NAME


def open_camera():
    if hasattr(cv2, "CAP_DSHOW"):
        capture = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if capture.isOpened():
            return capture

    return cv2.VideoCapture(CAMERA_INDEX)


def main():
    cap = open_camera()

    if not cap.isOpened():
        raise RuntimeError("Camera is not opening. Check CAMERA_INDEX and camera permissions.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame read error.")
                break

            frame = cv2.flip(frame, 1)

            cv2.putText(
                frame,
                "AirMouse camera loop - press Q to quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
