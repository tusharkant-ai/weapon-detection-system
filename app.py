import streamlit as st
from ultralytics import YOLO
import cv2
from datetime import datetime
import tempfile
import time
import base64

# ============================
# Load YOLOv8 Model
# ============================
# ============================
# Load YOLOv8 Model
# ============================
MODEL_PATH = "runs/weights/best.pt"

try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    st.error(f"❌ Model not found at {MODEL_PATH}")
    st.info("👉 Please make sure 'best.pt' is placed inside the 'runs/' folder.")
    st.stop()
# ============================
# Streamlit UI Config
# ============================
st.set_page_config(page_title="Weapon Detection Dashboard", layout="wide")
st.title("🔴 Weapon Detection Dashboard – YOLOv8")

# ============================
# Alert Toggle
# ============================
alert_enabled = st.toggle("🔔 Enable On-Screen Alerts")
st.markdown("---")

# ============================
# Create Tabs
# ============================
tab1, tab2 = st.tabs(["📹 Live Camera", "📤 Upload Image/Video"])

# ============================
# CAMERA TAB
# ============================
with tab1:
    st.subheader("📹 Live Camera Feed")
    start_camera = st.toggle("▶️ Start Live Detection", key="live")
    FRAME_WINDOW = st.image([], use_container_width=True)

    # Initialize session states
    if 'alert_active' not in st.session_state:
        st.session_state['alert_active'] = False
    if 'sound_played' not in st.session_state:
        st.session_state['sound_played'] = False
    if 'alert_cooldown' not in st.session_state:
        st.session_state['alert_cooldown'] = 0

    # Dynamic placeholders
    alert_placeholder = st.empty()
    audio_placeholder = st.empty()  # ✅ for dynamic audio control

    # Function to encode alert sound
    def get_audio_base64(file_path):
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    alert_sound_path = "alert_sound.mp3"
    try:
        alert_audio_base64 = get_audio_base64(alert_sound_path)
    except:
        alert_audio_base64 = None
        st.warning("⚠️ Alert sound file not found! Place 'alert_sound.mp3' in the same folder.")

    if start_camera:
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Unable to access the camera.")
                break

            results = model(frame)
            annotated_frame = results[0].plot()

            # Detect all classes except 'person'
            detected_weapons = [
                (model.names[int(box.cls[0])].strip().lower(), float(box.conf[0]))
                for box in results[0].boxes
                if model.names[int(box.cls[0])].strip().lower() != "person"
            ]

            weapon_detected = len(detected_weapons) > 0
            current_time = time.time()

            # ALERT LOGIC
            if alert_enabled and weapon_detected:
                alert_label, conf = detected_weapons[0]
                timestamp = datetime.now().strftime("%H:%M:%S")

                if not st.session_state['alert_active']:
                    st.session_state['alert_active'] = True
                    st.session_state['sound_played'] = False
                    st.session_state['alert_cooldown'] = current_time

                # Show visual alert
                alert_placeholder.markdown(f"""
                <div style="
                    position: fixed;
                    top: 90px;
                    right: 50px;
                    background: linear-gradient(135deg, #ff4b4b, #ff0000);
                    color: white;
                    padding: 16px 25px;
                    border-radius: 15px;
                    box-shadow: 0px 4px 20px rgba(255,0,0,0.5);
                    font-size: 18px;
                    font-weight: bold;
                    opacity: 1;
                    z-index: 9999;
                    text-align:center;">
                    ⚠️ WEAPON DETECTED! <br>
                    <small>{alert_label.upper()} ({conf*100:.1f}%) — {timestamp}</small>
                </div>
                """, unsafe_allow_html=True)

                # ✅ Play sound once, managed by audio_placeholder
                if (not st.session_state['sound_played']) and alert_audio_base64:
                    audio_placeholder.markdown(
                        f"""
                        <audio autoplay>
                            <source src="data:audio/mp3;base64,{alert_audio_base64}" type="audio/mp3">
                        </audio>
                        """,
                        unsafe_allow_html=True
                    )
                    st.session_state['sound_played'] = True

            else:
                # No weapon detected — fade out alert and stop sound
                if st.session_state['alert_active']:
                    if current_time - st.session_state['alert_cooldown'] > 1:
                        st.session_state['alert_active'] = False
                        st.session_state['sound_played'] = False

                        # ✅ Stop the sound
                        audio_placeholder.empty()

                        # Fade out alert visually
                        alert_placeholder.markdown("""
                        <div style="
                            position: fixed;
                            top: 90px;
                            right: 50px;
                            background: linear-gradient(135deg, #ff4b4b, #ff0000);
                            color: white;
                            padding: 16px 25px;
                            border-radius: 15px;
                            box-shadow: 0px 4px 20px rgba(255,0,0,0.5);
                            font-size: 18px;
                            font-weight: bold;
                            opacity: 0;
                            transition: opacity 1s ease-out;
                            z-index: 9999;
                            text-align:center;">
                            ⚠️ WEAPON DETECTED! (Cleared)
                        </div>
                        """, unsafe_allow_html=True)
                        time.sleep(1)
                        alert_placeholder.empty()

            # ✅ FIXED COLOR for live preview
            FRAME_WINDOW.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
            time.sleep(0.05)

        cap.release()
        
        
# ============================
# UPLOAD TAB
# ============================
with tab2:
    st.subheader("📤 Upload Image / Video")
    
    # Store uploaded file in session state to persist across reruns
    if "uploaded_file_path" not in st.session_state:
        st.session_state.uploaded_file_path = None
        st.session_state.is_video = False

    uploaded_file = st.file_uploader("Choose an image or video file", type=["jpg", "png", "mp4", "avi"])

    if uploaded_file:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())
        st.session_state.uploaded_file_path = temp_file.name
        st.session_state.is_video = uploaded_file.type.startswith("video")

    # Preview size slider
    preview_size = st.slider("🔍 Adjust Preview Size (%)", min_value=20, max_value=100, value=70)
    preview_width = int(preview_size * 10)

    # Frame rate slider (only if video uploaded)
    if st.session_state.is_video:
        if "fps" not in st.session_state:
            st.session_state.fps = 15  # default fps
        st.session_state.fps = st.slider("🎞️ Select Processing Frame Rate (fps)", min_value=1, max_value=30, value=st.session_state.fps)

    # Show Process button only if a file is uploaded
    if st.session_state.uploaded_file_path:
        st.markdown("""
            <style>
            div.stButton > button:first-child {
                background: linear-gradient(135deg, #ff4b4b, #ff0000);
                color: white;
                font-weight: bold;
                border-radius: 12px;
                padding: 10px 25px;
                font-size: 16px;
                transition: transform 0.2s ease;
            }
            div.stButton > button:first-child:hover {
                background: linear-gradient(135deg, #ff0000, #ff1a1a);
                transform: scale(1.05);
            }
            </style>
        """, unsafe_allow_html=True)
        
        process_btn = st.button("▶️ Process & Preview")

        if process_btn:
            file_path = st.session_state.uploaded_file_path

            # ========== IMAGE PROCESSING ==========
            if not st.session_state.is_video:
                img = cv2.imread(file_path)
                results = model(img)
                annotated = results[0].plot()
                st.image(annotated, channels="BGR", width=preview_width, output_format="JPEG")

                st.download_button(
                    "⬇️ Download Processed Image",
                    data=cv2.imencode('.jpg', annotated)[1].tobytes(),
                    file_name="detected_image.jpg",
                    key="image_download"
                )

            # ========== VIDEO PROCESSING ==========
            else:
                cap = cv2.VideoCapture(file_path)
                video_fps = cap.get(cv2.CAP_PROP_FPS)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_path = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                out = cv2.VideoWriter(output_path, fourcc, st.session_state.fps, (int(cap.get(3)), int(cap.get(4))))
                
                st_frame = st.image([], width=preview_width)
                frame_count = 0
                frame_interval = max(int(video_fps / st.session_state.fps), 1)

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    if frame_count % frame_interval == 0:
                        results = model(frame)
                        annotated = results[0].plot()

                        # ✅ keep BGR for saving
                        out.write(annotated)

                        # ✅ convert to RGB for correct Streamlit display
                        rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        st_frame.image(rgb_frame, width=preview_width)
                    frame_count += 1

                cap.release()
                out.release()

                st.download_button(
                    "⬇️ Download Processed Video",
                    data=open(output_path, "rb").read(),
                    file_name="detected_video.mp4",
                    key="video_download"
                )