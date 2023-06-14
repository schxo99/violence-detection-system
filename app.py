from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import torch
import numpy as np
import datetime
from models.experimental import attempt_load
from utils.general import non_max_suppression
from utils.dataloaders import letterbox
from utils.plots import plot_one_box
import os

# YOLOv5 모델 로드
weights = 'weights/best.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = attempt_load(weights, device=device)
imgsz = 640

# 클래스 이름 및 색상 정의
names = model.module.names if hasattr(model, 'module') else model.names
colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

# Flask 앱 객체 생성
app = Flask(__name__)

# SocketIO 객체 생성 및 Flask 앱과 연결
socketio = SocketIO(app)

# 웹캠 비디오 스트리밍 및 객체 감지
def detect_objects():
    # 웹캠 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(0)

    while True:
        # 비디오 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 회전
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # 프레임 크기 조정 및 전처리
        img = letterbox(frame, new_shape=imgsz)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img)

        # 객체 감지
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 객체 감지 결과 가져오기
        pred = model(img, augment=False)[0]
        pred = non_max_suppression(pred, 0.8, 0.5)

        # 결과 시각화 및 사진 캡처
        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, frame, label=label, color=colors[int(cls)])

                    # 객체가 감지되면 이미지를 저장합니다
                    if names[int(cls)] == 'violence':
                        # 경계 상자를 확장한 좌표 계산
                        expanded_box = [int(xyxy[0]) - 20, int(xyxy[1]) - 20, int(xyxy[2]) + 20, int(xyxy[3]) + 20]
                        # 확장한 좌표를 이미지 경계 내에 조정
                        expanded_box[0] = max(0, expanded_box[0])
                        expanded_box[1] = max(0, expanded_box[1])
                        expanded_box[2] = min(frame.shape[1], expanded_box[2])
                        expanded_box[3] = min(frame.shape[0], expanded_box[3])
                        # 경계 상자를 포함한 이미지 저장
                        object_img = frame[expanded_box[1]:expanded_box[3], expanded_box[0]:expanded_box[2]]
                        now = datetime.datetime.now()
                        timestamp = now.strftime("%y%m%d_%H%M")
                        img_name = f"capture_{timestamp}_{int(xyxy[0])}_{int(xyxy[1])}.jpg"
                        save_dir = 'static/captures'
                        img_path = os.path.join(save_dir, img_name)
                        # 이미지를 더 크게 조정하여 저장
                        scale_percent = 120  # 이미지를 120%로 확대
                        width = int(object_img.shape[1] * scale_percent / 100)
                        height = int(object_img.shape[0] * scale_percent / 100)
                        object_img = cv2.resize(object_img, (width, height))
                        cv2.imwrite(img_path, object_img)
                        print("이미지 저장")

                         # 객체 감지 결과를 클라이언트에게 전송
                        detection_result = {
                            'image_url': img_path,
                            'image_info': f"Object detected: {label}, Image path: {img_path}"
                        }
                        socketio.emit('detection_result', detection_result, namespace='/')
                        
                        # 로그 메시지를 생성하여 클라이언트로 전성
                        now = datetime.datetime.now()
                        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                        log_message = f"폭력이 감지되었습니다. {timestamp}"
                        socketio.emit('log_message', log_message, namespace='/')

        # 프레임 출력
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    if ratio_pad is None: 
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain

    return boxes

@app.route('/')
def index():
    image_files = os.listdir('static/captures')
    if image_files:
        last_image_file = image_files[-1]  # 마지막 이미지 파일 선택
        image_path = os.path.join('static/captures', last_image_file)
        image_info = last_image_file[:-4]  # 이미지 파일 이름에서 확장자(.jpg) 제거
        return render_template('main.html', image_url=image_path, image_info=image_info)
    else:
        return render_template('main.html', image_url=None, image_info=None)

@app.route('/video')
def video_feed():
    return Response(detect_objects(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    print("야발")
    image_files = os.listdir('static/captures')
    return render_template('capture.html', images=image_files)
@socketio.on('log_message')
def handle_log_message(message):
    emit('log_message', message)

if __name__ == '__main__':
    socketio.run(app, debug=True)