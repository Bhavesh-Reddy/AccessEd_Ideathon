import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
import threading

# --- GLOBALS FOR FLASK ACCESS ---
last_prediction = "Waiting..."
last_confidence = 0.0
data_lock = threading.Lock() # Protects variables during updates
cap = None
cap_lock = threading.Lock() # Protects camera during release

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

def get_asl_data():
    """Returns current prediction data (thread-safe)"""
    with data_lock:
        return {
            "label": last_prediction,
            "confidence": round(last_confidence, 2)
        }

def release_asl_camera():
    """Forces the ASL camera to release resources."""
    global cap
    with cap_lock:
        if cap is not None:
            try:
                if cap.isOpened():
                    cap.release()
                print("[ASL] Camera released.")
            except Exception as e:
                print(f"[ASL] Error releasing: {e}")
        cap = None

def asl_generator(model_path, active_check):
    global last_prediction, last_confidence, cap
    
    # 1. LOADING FRAME
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "LOADING AI...", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    _, buf = cv2.imencode('.jpg', blank)
    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    # 2. LOAD MODEL
    model = None
    try:
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            model = data['model'] if isinstance(data, dict) else data
        print("[ASL] Model Loaded.")
    except Exception as e:
        print(f"[ASL] Model Error: {e}")
        # Use demo mode if model fails (prevents crash)
        model = None 

    # 3. CAMERA SETUP
    with cap_lock:
        if cap is not None: cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    
    if cap is None or not cap.isOpened():
        print("[ASL] No camera.")
        return

    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    
    # 4. MAIN LOOP
    try:
        while active_check():
            with cap_lock:
                if cap is None or not cap.isOpened(): break
                success, frame = cap.read()
            
            if not success:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            
            current_pred = "..."
            current_conf = 0.0

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    if model:
                        try:
                            # Extract 3D coordinates (x, y, z)
                            data_aux = []
                            for lm in hand_landmarks.landmark:
                                data_aux.extend([lm.x, lm.y, lm.z])

                            # Prediction
                            prediction = model.predict([data_aux])
                            proba = model.predict_proba([data_aux])
                            
                            current_pred = str(prediction[0])
                            current_conf = float(np.max(proba))
                            
                            # Filter low confidence
                            if current_conf > 0.5:
                                with data_lock:
                                    last_prediction = current_pred
                                    last_confidence = current_conf
                            else:
                                current_pred = "Uncertain"

                        except Exception as e:
                            # Handle dimension mismatch (e.g. model trained on 2D vs 3D)
                            print(f"Pred Error: {e}")
            
            # Visualization
            cv2.rectangle(frame, (0, 400), (640, 480), (0, 0, 0), -1)
            
            # Color logic
            color = (0, 255, 0) if current_conf > 0.7 else (0, 255, 255)
            
            cv2.putText(frame, f"Sign: {current_pred}", (20, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            
            # Confidence bar
            bar_w = int(current_conf * 200)
            cv2.rectangle(frame, (400, 430), (400 + bar_w, 450), color, -1)
            cv2.rectangle(frame, (400, 430), (600, 450), (255, 255, 255), 2)

            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    except Exception as e:
        print(f"[ASL] Loop Error: {e}")

    finally:
        hands.close()
        release_asl_camera()
        
        off = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buf = cv2.imencode('.jpg', off)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")