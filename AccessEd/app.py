import os
import time
import queue
import threading
import torch
import pyttsx3
import pythoncom
import speech_recognition as sr
from flask import Flask, render_template_string, request, redirect, jsonify, Response, session
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# --- IMPORT MODULES ---
try:
    from asl import asl_generator, release_asl_camera, get_asl_data
    print("✓ ASL module imported")
except ImportError as e:
    print(f"⚠ ASL module missing: {e}")
    def asl_generator(*args): yield b''
    def release_asl_camera(): pass
    def get_asl_data(): return {"label": "...", "confidence": 0.0}

try:
    from cursor import cursor_generator, release_camera as release_cursor_camera
    print("✓ Cursor module imported")
except ImportError as e:
    print(f"⚠ Cursor module missing: {e}")
    def cursor_generator(*args): yield b''
    def release_cursor_camera(): pass

# ---------------- CONFIGURATION ----------------
app = Flask(__name__)
app.secret_key = "accessed_secure_key_2026"

# UPDATE THIS PATH TO YOUR ACTUAL MODEL PATH
MISTRAL_PATH = r"C:\Users\GOPAL\.cache\huggingface\hub\models--mistralai--Mistral-7B-Instruct-v0.2\snapshots\63a8b081895390a26e140280378bc85ec8bce07a"

# ---------------- GLOBAL STATE ----------------
active = {"asl": False, "cursor": False}
chat_model = None
chat_tokenizer = None
chat_pipeline = None
model_status = "Initializing AI..."

active_lock = threading.Lock()
camera_lock = threading.Lock()
current_camera_user = None

# Speech Components
speech_recognizer = sr.Recognizer()
tts_queue = queue.Queue()

users_db = {"admin": {"password": "123", "disability": "None"}}

# ---------------- LOAD CHATBOT ----------------
def load_chatbot():
    global chat_model, chat_tokenizer, chat_pipeline, model_status
    print("🔄 [AI] Loading Mistral-7B...")
    try:
        chat_tokenizer = AutoTokenizer.from_pretrained(MISTRAL_PATH, local_files_only=True)
        chat_tokenizer.pad_token = chat_tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        chat_model = AutoModelForCausalLM.from_pretrained(
            MISTRAL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            local_files_only=True,
            low_cpu_mem_usage=True
        )

        chat_pipeline = pipeline(
            "text-generation",
            model=chat_model,
            tokenizer=chat_tokenizer,
            device_map="auto"
        )
        
        model_status = "Ready"
        print("✅ [AI] Mistral Loaded Successfully")
        
    except Exception as e:
        model_status = f"Error: {str(e)}"
        print(f"❌ [AI] Load Error: {e}")

def tts_worker():
    print("✅ [TTS] Worker started")
    while True:
        try:
            # 1. Get text from queue (blocks until data arrives)
            text = tts_queue.get()
            if text is None: break
            
            print(f"🔊 [TTS] Speaking: {text[:20]}...")

            # 2. Initialize Windows COM for this specific task
            pythoncom.CoInitialize()

            # 3. Create a FRESH engine instance
            engine = pyttsx3.init()
            engine.setProperty("rate", 175)
            
            # Select voice (optional safety check)
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)

            # 4. Speak and BLOCK until finished
            engine.say(text)
            engine.runAndWait()
            
            # 5. Clean up manually to free resources
            engine.stop()
            del engine

        except Exception as e:
            print(f"❌ [TTS] Error: {e}")
        
        finally:
            # 6. Uninitialize Windows COM (Critical!)
            try:
                pythoncom.CoUninitialize()
            except:
                pass
            
            # 7. Mark task as done so the queue proceeds
            tts_queue.task_done()
        
# ---------------- CAMERA HELPER ----------------
def force_release_all_cameras():
    global current_camera_user
    with camera_lock:
        print("🔄 [SYSTEM] Releasing cameras...")
        release_asl_camera()
        release_cursor_camera()
        time.sleep(0.5)  # Give time for camera to release
        current_camera_user = None

# Start background threads
threading.Thread(target=load_chatbot, daemon=True).start()
threading.Thread(target=tts_worker, daemon=True).start()

# ---------------- LOAD HTML TEMPLATES ----------------
def load_template(filepath):
    """Load HTML template with proper UTF-8 encoding"""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return None
    return None

INDEX_HTML = load_template('templates/index.html') or """
<!DOCTYPE html>
<html><head><title>AccessEd Login</title></head>
<body><h1>Login/Signup</h1>
<form action="/login" method="post">
    <input name="name" placeholder="Username" required>
    <input name="password" type="password" placeholder="Password" required>
    <button type="submit">Login</button>
</form>
<form action="/signup" method="post">
    <input name="name" placeholder="Username" required>
    <input name="password" type="password" placeholder="Password" required>
    <input name="disability" placeholder="Disability Type">
    <button type="submit">Signup</button>
</form>
</body></html>
"""

DASHBOARD_HTML = load_template('templates/dashboard.html')

# ---------------- ROUTES ----------------
@app.route("/")
def index():
    user = session.get("name")
    if user and user in users_db: 
        return redirect("/dashboard")
    session.clear()
    return render_template_string(INDEX_HTML)

@app.route("/login", methods=["POST"])
def login():
    data = request.form
    name, password = data.get("name"), data.get("password")
    if name in users_db and users_db[name]["password"] == password:
        session["name"] = name
        session["disability"] = users_db[name]["disability"]
        return jsonify({"success": True, "redirect": "/dashboard"})
    return jsonify({"error": "Invalid credentials"}), 401

@app.route("/signup", methods=["POST"])
def signup():
    data = request.form
    name = data.get("name")
    if name in users_db: 
        return jsonify({"error": "User exists"}), 409
    users_db[name] = {
        "password": data.get("password"), 
        "disability": data.get("disability", "None")
    }
    session["name"] = name
    session["disability"] = data.get("disability", "None")
    return jsonify({"success": True, "redirect": "/dashboard"})

@app.route("/dashboard")
def dashboard():
    if "name" not in session: 
        return redirect("/")
    
    if DASHBOARD_HTML:
        return render_template_string(
            DASHBOARD_HTML, 
            name=session["name"], 
            disability=session.get("disability", "None"),
            status=model_status
        )
    else:
        return "<h1>Dashboard</h1><p>Template not found</p>"

@app.route("/logout")
def logout():
    force_release_all_cameras()
    with active_lock:
        for key in active: 
            active[key] = False
    session.clear()
    return redirect("/")

# ---------------- CHAT API ----------------
@app.route("/chat/listen", methods=["POST"])
def chat_listen():
    try:
        with sr.Microphone() as source:
            speech_recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = speech_recognizer.listen(source, timeout=5, phrase_time_limit=10)
        text = speech_recognizer.recognize_google(audio)
        return jsonify({"status": "success", "text": text})
    except sr.WaitTimeoutError:
        return jsonify({"status": "error", "message": "No speech detected"})
    except sr.UnknownValueError:
        return jsonify({"status": "error", "message": "Could not understand audio"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route("/chat/send", methods=["POST"])
def chat_send():
    if chat_pipeline is None: 
        return jsonify({"response": "AI is still initializing. Please wait..."})
    
    user_msg = request.json.get("message", "").strip()
    if not user_msg:
        return jsonify({"response": "Please provide a message."})
    
    try:
        prompt = f"<s>[INST] You are a helpful AI assistant for students with disabilities. Be concise and supportive. {user_msg} [/INST]"
        sequences = chat_pipeline(
            prompt, 
            max_new_tokens=150, 
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        response_text = sequences[0]['generated_text'].split("[/INST]")[-1].strip()
        return jsonify({"response": response_text})
    except Exception as e:
        print(f"Chat error: {e}")
        return jsonify({"response": f"I encountered an error: {str(e)}"})

@app.route("/chat/speak", methods=["POST"])
def chat_speak():
    text = request.json.get("text", "")
    if text: 
        tts_queue.put(text)
    return jsonify({"status": "ok"})

# ---------------- FEATURE CONTROL ----------------
@app.route("/launch/<feature>")
def launch(feature):
    global current_camera_user
    
    if feature not in active: 
        return jsonify({"error": "Unknown feature"}), 400

    print(f"\n[SYSTEM] Launch Request: {feature.upper()}")

    with active_lock:
        # 1. Flag everything as stopped immediately
        for key in active: 
            active[key] = False
        
        # 2. Release hardware
        force_release_all_cameras()
        
        # 3. Wait for previous thread to actually die (CRITICAL)
        print("[SYSTEM] Waiting for cleanup...")
        time.sleep(1.5) 
        
        # 4. Activate new feature
        active[feature] = True
        current_camera_user = feature
    
    print(f"✅ [SYSTEM] Launched {feature}")
    return jsonify({"status": "started", "feature": feature})

@app.route("/stop/<feature>")
def stop(feature):
    print(f"[SYSTEM] Stop Request: {feature}")
    # Immediately flag false to break the loops in cursor.py/asl.py
    if feature in active:
        active[feature] = False
    
    # Run cleanup in a separate thread so it doesn't block the request
    threading.Thread(target=force_release_all_cameras).start()
    
    return jsonify({"status": "stopped", "feature": feature})

# ---------------- VIDEO FEEDS ----------------
@app.route("/video_feed/asl")
def video_asl():
    def active_check():
        return active.get("asl", False)
    
    return Response(
        asl_generator("gesture_model.pkl", active_check), 
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/video_feed/cursor")
def video_cursor():
    def active_check():
        return active.get("cursor", False)
    
    return Response(
        cursor_generator(active_check), 
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# ---------------- ASL DATA API ----------------
@app.route('/asl/label')
def asl_label():
    """Returns current ASL prediction and confidence"""
    data = get_asl_data()
    return jsonify(data)

# ---------------- STATUS ENDPOINTS ----------------
@app.route('/status/cursor')
def cursor_status_route():
    return jsonify({'active': active['cursor']})

@app.route('/status/system')
def system_status():
    return jsonify({
        'ai_model': model_status, 
        'active_features': active,
        'camera_user': current_camera_user
    })

if __name__ == "__main__":
    print("🚀 Starting AccessEd Server...")
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)