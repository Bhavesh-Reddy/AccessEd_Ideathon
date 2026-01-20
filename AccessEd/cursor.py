import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import threading

# Configuration
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0 

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

# --- GLOBALS FOR EXTERNAL ACCESS ---
cursor_status = "IDLE"
last_click_at = 0
cap = None  # Global camera object to allow force-release
cap_lock = threading.Lock() # CRITICAL: Prevents race conditions

def release_camera():
    """Forces the camera to release resources."""
    global cap, cursor_status
    with cap_lock:
        if cap is not None:
            try:
                if cap.isOpened():
                    cap.release()
                print("[CURSOR] Camera released.")
            except Exception as e:
                print(f"[CURSOR] Error releasing camera: {e}")
        cap = None
        cursor_status = "STOPPED"

def cursor_generator(active_check):
    global cap, cursor_status, last_click_at
    
    print(f"[CURSOR] Generator started")
    
    # 1. IMMEDIATE FEEDBACK FRAME
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(blank, "STARTING CURSOR CONTROL...", (120, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    _, buf = cv2.imencode('.jpg', blank)
    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    # 2. WAIT FOR ACTIVE STATUS
    timeout = 5
    start_wait = time.time()
    while not active_check() and (time.time() - start_wait) < timeout:
        time.sleep(0.1)
    
    if not active_check():
        return

    # 3. CAMERA SETUP
    with cap_lock:
        if cap is not None:
            cap.release()
        
        # Try index 0, then 1
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW is faster on Windows
        if not cap.isOpened():
            cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

        if not cap.isOpened():
            print("[CURSOR] ❌ No camera available")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("[CURSOR] ✅ Camera opened successfully")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    screen_w, screen_h = pyautogui.size()
    prev_x, prev_y = screen_w // 2, screen_h // 2
    smoothening = 5
    last_click_time = 0
    frame_count = 0
    
    try:
        while active_check():
            # Thread-safe camera read
            with cap_lock:
                if cap is None or not cap.isOpened():
                    break
                success, frame = cap.read()
            
            if not success:
                time.sleep(0.01)
                continue

            frame_count += 1
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            status_text = "NO HAND"
            status_color = (0, 165, 255) # Orange

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                idx_tip = lm.landmark[8]
                thm_tip = lm.landmark[4]

                # --- MOVE CURSOR ---
                # Map hand (normalized 0-1) to screen coordinates
                # We use a smaller window (100px margin) for easier reach
                margin = 100
                x_mapped = np.interp(idx_tip.x * w, (margin, w - margin), (0, screen_w))
                y_mapped = np.interp(idx_tip.y * h, (margin, h - margin), (0, screen_h))

                curr_x = prev_x + (x_mapped - prev_x) / smoothening
                curr_y = prev_y + (y_mapped - prev_y) / smoothening

                try:
                    pyautogui.moveTo(int(curr_x), int(curr_y), _pause=False)
                    prev_x, prev_y = curr_x, curr_y
                    status_text = "TRACKING"
                    status_color = (0, 255, 0)
                except: pass

                # --- CLICK DETECTION (Pinch) ---
                # Calculate distance between thumb and index
                dist_px = np.hypot((idx_tip.x - thm_tip.x)*w, (idx_tip.y - thm_tip.y)*h)
                
                cx, cy = int(idx_tip.x * w), int(idx_tip.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

                if dist_px < 30: # Threshold in pixels
                    status_text = "CLICKING"
                    status_color = (0, 0, 255)
                    cv2.circle(frame, (cx, cy), 15, (0, 0, 255), 2)
                    
                    if (time.time() - last_click_time) > 0.4:
                        pyautogui.click(_pause=False)
                        last_click_time = time.time()
                        last_click_at = last_click_time

                # Draw landmarks
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

            cursor_status = status_text
            
            # UI Overlay
            cv2.rectangle(frame, (100, 100), (w-100, h-100), (255, 255, 0), 1)
            cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

            # High quality JPEG for better stream
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

    except Exception as e:
        print(f"[CURSOR] Error: {e}")

    finally:
        hands.close()
        release_camera()
        
        # Send OFF frame
        off = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(off, "STOPPED", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buf = cv2.imencode('.jpg', off)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")