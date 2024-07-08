'''
bot_face_track_recognizer.py

カメラ映像を取得し、顔を検出して認識する顔追跡システムのボット用スクリプトです
カメラで顔を検出し、顔の特徴を抽出して辞書と比較し、顔認識を行います
また、顔の中心を捉えてカメラのパンとチルトを制御し、顔の追跡も行います
'''

import cv2
import numpy as np
import time
from pathlib import Path
from collections import Counter
from picamera2 import Picamera2
# from bot_motor_controller import pan_tilt, neopixels_all, neopixels_off

class Camera():
    def __init__(self):
        # self.cap = cv2.VideoCapture(0)
        # self.cap.set(3, 640)
        # self.cap.set(4, 480)
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"format": "RGB888"}))
        self.picam2.start()

    def get_frame(self):
        # ret, frame = self.cap.read()
        frame = self.picam2.capture_array()
        if frame is not None:
            return frame
        else:
            print("🖥️ SYSTEM: カメラからのフレーム取得に失敗しました。")
            return None

    def release_camera(self):
        self.picam2.stop()
        self.picam2.close()


def face_recognize():
    recognized_faces_set = set()
    # 加载模型
    face_detector_weights = str(Path("dnn_models/face_detection_yunet_2023mar.onnx").resolve())
    face_detector = cv2.FaceDetectorYN_create(face_detector_weights, "", (0, 0))

    face_recognizer_weights = str(Path("dnn_models/face_recognizer_fast.onnx").resolve())
    face_recognizer = cv2.FaceRecognizerSF_create(face_recognizer_weights, "")

    COSINE_THRESHOLD = 0.363

    dictionary = []
    files = Path("face_dataset").glob("*.npy")
    for file in files:
        feature = np.load(file)
        user_id = Path(file).stem
        dictionary.append((user_id, feature))

    def match(recognizer, feature1, data_directory):
        for element in data_directory:
            user_id, feature2 = element
            score = recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
            if score > COSINE_THRESHOLD:
                return user_id, score
        return None

    recognized_faces = []

    cam = Camera()

    while True:
        frame = cam.get_frame()
        if frame is None:
            break

        height, width, _ = frame.shape
        face_detector.setInputSize((width, height))

        _, faces = face_detector.detect(frame)

        if faces is not None:
            for face in faces:
                x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    continue
                
                roi = frame[y:y+h, x:x+w]
                
                if roi.size == 0:
                    continue
                
                face_feature = face_recognizer.feature(roi)

                user = match(face_recognizer, face_feature, dictionary)
                user_id = user[0] if user is not None else 'unknown'

                # 将当前识别到的用户ID添加到集合中
                recognized_faces_set.add(user_id)


                # 在框中绘制矩形和识别结果
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = str(user_id)
                cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("face detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cam.release_camera()
    cv2.destroyAllWindows()

    return list(recognized_faces_set), len(recognized_faces_set)

if __name__ == '__main__':
    recognized_faces, face_count = face_recognize()
    print(f"Recognized faces: {recognized_faces}")
    print(f"Number of faces detected: {face_count}")