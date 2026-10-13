# 絵を空中に描くプログラム（正方形ガイド枠＆ログ保存機能追加版）

import cv2
import mediapipe as mp
import math
import numpy as np
import time
import torch
import torch.nn as nn
import datetime # 日時を記録するために追加

# ========== AIモデルの設定 ==========
classes = ["clock", "cat", "apple", "cake", "boomerang"]
num_classes = len(classes)

class QuickDrawCNN(nn.Module):
    def __init__(self):
        super(QuickDrawCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# ========== 設定 ==========
PINCH_THRESHOLD = 0.02
# ==========================

def map_coordinate(val, min_val, max_val, out_max):
    if max_val - min_val == 0:
        return 0
    norm_val = (val - min_val) / (max_val - min_val)
    norm_val = max(0.0, min(1.0, norm_val))
    return int(norm_val * out_max)

def main():
    print("AIモデルを読み込んでいます...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = QuickDrawCNN().to(device)
    model.load_state_dict(torch.load("quickdraw_cnn.pth", map_location=device, weights_only=True))
    model.eval() 
    print(f"AIモデルの準備完了！（使用デバイス: {device}）")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils 
    
    hands = mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
        max_num_hands=1
    )

    cap = cv2.VideoCapture(0) 
    
    ret = False
    for i in range(10): 
        ret, frame = cap.read()
        if ret:
            break 
        time.sleep(0.1) 
        
    if not ret:
        print("カメラが見つかりません。")
        return
        
    h, w, c = frame.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    print("=== 操作方法 ===")
    print("・親指と人差し指を近づけると線が描けます（お絵描きモード時）")
    print("・'A' キー：AIにお題を予想させる")
    print("・'O' キー：予想が【正解】だった場合にログを保存 ★NEW")
    print("・'X' キー：予想が【不正解】だった場合にログを保存 ★NEW")
    print("・'R' キー：キャリブレーションを開始 / やり直し")
    print("・'C' キー：キャンバスを消去（クリア）")
    print("・'F' キー：プログラム終了")
    print("・お題：clock, cat, apple, cake, boomerang")
    print("================")

    # === 【NEW】お絵描きガイド枠（正方形）の計算 ===
    # 画面の短い方の辺の 70% を枠のサイズにする
    box_size = int(min(h, w) * 0.7)
    # 画面の中央に配置するための座標
    box_x1 = (w - box_size) // 2
    box_y1 = (h - box_size) // 2
    box_x2 = box_x1 + box_size
    box_y2 = box_y1 + box_size
    # ===============================================

    is_calibrating = True  
    calib_step = 0         
    calib_points = []      
    was_pinching = False   
    last_calib_time = 0.0  

    map_min_x, map_max_x = 0.2, 0.8
    map_min_y, map_max_y = 0.2, 0.8

    targets = [
        (int(w * 0.1), int(h * 0.1)),
        (int(w * 0.9), int(h * 0.1)),
        (int(w * 0.1), int(h * 0.9)),
        (int(w * 0.9), int(h * 0.9))
    ]

    prev_x, prev_y = 0, 0
    prediction_text = ""
    last_prediction = "" # 【NEW】最後に予想したお題を記録しておく変数

    # === フルスクリーン（全画面）設定 ===
    cv2.namedWindow('AI Drawing MVP (Projector)', cv2.WINDOW_NORMAL)
    # ==========================================

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        # プロジェクター表示用
        display_image = canvas.copy()
        admin_frame = frame.copy()

        # === ガイド枠の描画 ===
        if not is_calibrating:
            cv2.rectangle(display_image, (box_x1, box_y1), (box_x2, box_y2), (100, 100, 100), 5)
            cv2.putText(display_image, "Draw inside the box!", (box_x1, box_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 3)
        # ==============================

        if is_calibrating:
            current_min_x, current_max_x = 0.0, 1.0
            current_min_y, current_max_y = 0.0, 1.0
            
            target_x, target_y = targets[calib_step]
            cv2.circle(display_image, (target_x, target_y), 20, (255, 255, 0), 2)
            cv2.circle(display_image, (target_x, target_y), 5, (255, 255, 0), cv2.FILLED)
            cv2.putText(admin_frame, f"CALIBRATING... {calib_step}/4", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            current_min_x, current_max_x = map_min_x, map_max_x
            current_min_y, current_max_y = map_min_y, map_max_y
            
            # デバッグ用の青枠（体験者の仮想空間）
            cv2.rectangle(admin_frame, 
                          (int(current_min_x * w), int(current_min_y * h)), 
                          (int(current_max_x * w), int(current_max_y * h)), 
                          (255, 0, 0), 2)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(admin_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                mapped_landmarks = []
                for lm in hand_landmarks.landmark:
                    cx = map_coordinate(lm.x, current_min_x, current_max_x, w)
                    cy = map_coordinate(lm.y, current_min_y, current_max_y, h)
                    mapped_landmarks.append((cx, cy))
                
                if not is_calibrating:
                    for connection in mp_hands.HAND_CONNECTIONS:
                        p1 = mapped_landmarks[connection[0]]
                        p2 = mapped_landmarks[connection[1]]
                        cv2.line(display_image, p1, p2, (0, 255, 0), 5)
                    for p in mapped_landmarks:
                        cv2.circle(display_image, p, 7, (0, 0, 255), cv2.FILLED)

                thumb = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                pointer_x, pointer_y = mapped_landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                distance = math.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)
                is_pinching = distance < PINCH_THRESHOLD

                if is_pinching:
                    if is_calibrating:
                        current_time = time.time()
                        if not was_pinching and (current_time - last_calib_time > 1.0):
                            calib_points.append((index.x, index.y))
                            calib_step += 1
                            last_calib_time = current_time 
                            
                            if calib_step >= 4:
                                map_min_x = (calib_points[0][0] + calib_points[2][0]) / 2
                                map_max_x = (calib_points[1][0] + calib_points[3][0]) / 2
                                map_min_y = (calib_points[0][1] + calib_points[1][1]) / 2
                                map_max_y = (calib_points[2][1] + calib_points[3][1]) / 2
                                
                                if map_max_x <= map_min_x: map_max_x = map_min_x + 0.1
                                if map_max_y <= map_min_y: map_max_y = map_min_y + 0.1
                                
                                is_calibrating = False
                                canvas = np.zeros((h, w, 3), dtype=np.uint8) 
                    else:
                        cv2.circle(display_image, (pointer_x, pointer_y), 8, (0, 0, 255), cv2.FILLED)
                        if prev_x != 0 and prev_y != 0:
                            # 枠外描画の禁止はせず、どこでも描ける仕様のまま
                            cv2.line(canvas, (prev_x, prev_y), (pointer_x, pointer_y), (255, 255, 255), 15)
                        prev_x, prev_y = pointer_x, pointer_y
                else:
                    if not is_calibrating:
                        cv2.circle(display_image, (pointer_x, pointer_y), 12, (255, 255, 255), 2)
                    prev_x, prev_y = 0, 0

                was_pinching = is_pinching

        if prediction_text != "":
            cv2.putText(display_image, f"AI Guess: {prediction_text}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

        # === TVの16:9比率に合わせて左右（上下）に均等な黒枠を追加して中央配置 ===
        target_ratio = 16 / 9
        current_ratio = w / h

        if current_ratio < target_ratio:
            target_w = int(h * target_ratio)
            pad_w = target_w - w
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            display_image = cv2.copyMakeBorder(display_image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(50, 50, 50))
        elif current_ratio > target_ratio:
            target_h = int(w / target_ratio)
            pad_h = target_h - h
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            display_image = cv2.copyMakeBorder(display_image, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(50, 50, 50))
        # =========================================================================

        cv2.imshow('AI Drawing MVP (Projector)', display_image)
        cv2.imshow('Admin Camera View (MacBook)', admin_frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') or key == ord('C'):
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            prediction_text = "" 
            last_prediction = "" # 【NEW】クリア時にリセット
            
        elif key == ord('a') or key == ord('A'):
            # 画像配列の中から、ガイド枠の座標分だけをスライスで切り出し
            roi = canvas[box_y1:box_y2, box_x1:box_x2]
            
            # グレースケール変換 -> 28x28にリサイズ
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
            
            # AIへ渡す
            normalized = resized.astype(np.float32) / 255.0
            tensor = torch.tensor(normalized).unsqueeze(0).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(tensor)
                _, predicted_idx = torch.max(outputs, 1)
                
            prediction_text = classes[predicted_idx.item()]
            last_prediction = prediction_text # 【NEW】ログ用に記録
            print(f"AIの予想: {prediction_text}")
            
        # === 【NEW】ログ保存用の処理 ===
        elif key == ord('o') or key == ord('O'):
            if last_prediction != "":
                with open("ai_log.csv", "a", encoding="utf-8") as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{timestamp},{last_prediction},Correct\n")
                print(f"✅ ログ保存: {last_prediction} -> 正解(Correct)")
                last_prediction = "" # 連続保存防止

        elif key == ord('x') or key == ord('X'):
            if last_prediction != "":
                with open("ai_log.csv", "a", encoding="utf-8") as f:
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{timestamp},{last_prediction},Incorrect\n")
                print(f"❌ ログ保存: {last_prediction} -> 不正解(Incorrect)")
                last_prediction = "" # 連続保存防止
        # =================================

        elif key == ord('r') or key == ord('R'):
            is_calibrating = True
            calib_step = 0
            calib_points = []
            last_calib_time = time.time() 
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            prediction_text = ""
            last_prediction = "" # 【NEW】リセット時にリセット
            
        elif key == ord('f') or key == ord('F'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

if __name__ == "__main__":
    main()