import cv2
import time
from picamera2 import Picamera2
import os
import sys
import importlib.util
import numpy as np
import subprocess
import tkinter as tk
import sqlite3
from core_detection import detect_the_iris, extract_the_iris_features, normalize_the_iris
from load_dataset_util import find_best_match

#Initialize Database
base_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.normpath(os.path.join(base_dir, "..", "db_file", "stdnt_dbm.db"))

conn = sqlite3.connect(db_path)
cursor = conn.cursor()


def get_student_name(dataset_folder):
    cursor.execute("SELECT f_name, l_name FROM tbl_student_info WHERE stud_ID= ?", (dataset_folder,)) #get the name of the student based on the ID of the dataset
    result = cursor.fetchone()
    
    if result:
        return result[0]
    else:
        return "Unknown Student"
    

def live_iris_recognition(dataset_folder="dataset", screen_width=800):

    # Load dataset dynamically
    dataset_files = [f for f in os.listdir(dataset_folder) if f.endswith(".npy")]
    dataset_data = []
    dataset_labels = []

    for file in dataset_files:
        path = os.path.join(dataset_folder, file)
        features = np.load(path)
        features = (features > 0).astype(np.uint8)
        dataset_data.append(features)

        label = os.path.splitext(file)[0]
        if label.endswith("_norm"):
            label = label[:-5]
        dataset_labels.append(label)

    # Initialize Pi camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration(main={"size": (2304, 1296)}))
    picam2.start()
    picam2.set_controls({
        "AfMode": 2,
        "ExposureTime": 20000,
        "AnalogueGain": 4.0,
        "Sharpness": 2.0,
    })
    time.sleep(2)

    cascade_path = os.path.join(os.path.dirname(__file__), "haarcascade_eye.xml") #load Haarcascades
    eye_cascade = cv2.CascadeClassifier(cascade_path)
    if eye_cascade.empty():
        raise ValueError("Failed to load haarcascade_eye.xml")

    print("Press 'Q' to quit live recognition") #kill code

    # Main popup
    popup_root = tk.Tk()
    popup_root.withdraw()
    active_popup = None
    popup_shown_time = None

    def show_match_popup(label, match_rate, detection_time): #for match rate and detection time
        nonlocal active_popup, popup_shown_time
        if active_popup is not None:
            return

        active_popup = tk.Toplevel(popup_root)
        active_popup.title("Iris Match Detected") #display
        active_popup.geometry("300x150")
        active_popup.attributes("-topmost", True)

        tk.Label(active_popup, text=f"ID: {label}", font=("Arial", 14, "bold")).pack(pady=10)
        tk.Label(active_popup, text=f"Match Rate: {match_rate:.1f}%", font=("Arial", 12)).pack(pady=5)
        tk.Label(active_popup, text=f"Detection Time: {detection_time:.3f}s", font=("Arial", 12)).pack(pady=5)

        popup_shown_time = time.perf_counter()
        popup_root.update_idletasks()
        popup_root.update()

    # FALSE POSITIVE
    match_history = []
    history_size = 5
    min_hits_required = 3
    min_score = 0.92

    stable_identity = None #stable identity to lessen false acceptance, disabled since further testing is needed
    stable_counter = 0

    confirmed_label = None
    confirmed_score = 0.0

    while True:
        start_time = time.perf_counter()
        frame = picam2.capture_array()

        original_height, original_width = frame.shape[:2]

        frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        scale_ratio = screen_width / original_width
        frame_disp = cv2.resize(frame_disp, (screen_width, int(original_height * scale_ratio)))

        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_full = cv2.equalizeHist(gray_full)
        gray_full = cv2.GaussianBlur(gray_full, (3, 3), 0)

        eyes = eye_cascade.detectMultiScale(gray_full, scaleFactor=1.3, minNeighbors=5)

        captured_iris = None
        iris_label = "No Iris Match"
        match_score = 0.0
        detected_circle = None

        for (ex, ey, ew, eh) in eyes:
            eye_roi_gray = gray_full[ey:ey+eh, ex:ex+ew]
            detected_iris, circle = detect_the_iris(eye_roi_gray)

            if detected_iris is not None:
                iris_norm = normalize_the_iris(detected_iris, radial_res=64, angular_res=512)

                if iris_norm is not None:
                    iris_norm = cv2.equalizeHist(iris_norm)

                    captured_iris = frame[ey:ey+eh, ex:ex+ew].copy()
                    captured_iris_gray = cv2.cvtColor(captured_iris, cv2.COLOR_BGR2GRAY)
                    captured_iris_gray = cv2.cvtColor(captured_iris_gray, cv2.COLOR_GRAY2RGB)

                    live_features = extract_the_iris_features(iris_norm)

                    if live_features is not None:
                        live_features = (live_features > 0).astype(np.uint8)

                        iris_label, match_score = find_best_match(
                            live_features,
                            dataset_data,
                            dataset_labels
                        )

                    if circle is not None:
                        cx, cy, r = circle
                        detected_circle = (cx + ex, cy + ey, r)

                    break

        # FALSE POSITIVE FILTER, simple 
        if match_score >= min_score:
            match_history.append(iris_label)
        else:
            match_history.append("No Iris Match")

        if len(match_history) > history_size:
            match_history.pop(0)

        confirmed_label = None
        confirmed_score = 0.0

        if len(match_history) == history_size:

            best_label = max(set(match_history), key=match_history.count)
            count = match_history.count(best_label)

            if best_label != "No Iris Match" and count >= min_hits_required:

                if stable_identity == best_label:
                    stable_counter += 1
                else:
                    stable_identity = best_label
                    stable_counter = 1

                if stable_counter >= 2:
                    confirmed_label = stable_identity
                    confirmed_score = float(np.clip(match_score, 0.0, 1.0))

            else:
                stable_identity = None
                stable_counter = 0

        # POPUP
        if confirmed_label is not None and active_popup is None:
            student_name = get_student_name(confirmed_label)
            attendance_result=log_attendance(confirmed_label)
            show_match_popup(
                f"{student_name} ({confirmed_label})",
                float(confirmed_score) * 100.0,
                time.perf_counter() - start_time
            )
            tk.Label(active_popup,
                text=attendance_result,
                font=("Arial", 12, "bold"),
                fg="blue").pack(pady=5)

        if active_popup is not None and (time.perf_counter() - popup_shown_time) >= 3.0:
            active_popup.destroy()
            active_popup = None
            popup_shown_time = None

        popup_root.update_idletasks()
        popup_root.update()

        # UI OVERLAY for the cropped iris in the upper left part
        if captured_iris is not None:
            overlay_width = int(frame_disp.shape[1] * 0.10)
            h, w = captured_iris_gray.shape[:2]
            scale = overlay_width / w

            overlay_eye = cv2.resize(captured_iris_gray, (overlay_width, int(h * scale))) #check eye
            frame_disp[5:5+overlay_eye.shape[0], 10:10+overlay_eye.shape[1]] = overlay_eye

            text_y = 5 + overlay_eye.shape[0] + 25
            cv2.putText(frame_disp, f"ID: {iris_label}", (10, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            match_percent = float(np.clip(match_score, 0.0, 1.0)) * 100.0
            cv2.putText(frame_disp, f"Match Rate: {match_percent:.1f}%", (10, text_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame_disp, f"Detection Time: {time.perf_counter() - start_time:.3f}s",
                        (10, text_y + 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if detected_circle is not None:
            x, y, r = detected_circle
            x = int(x * scale_ratio)
            y = int(y * scale_ratio)
            r = int(r * scale_ratio)
            cv2.circle(frame_disp, (x, y), r, (0, 255, 0), 2)

        if confirmed_label is not None:
            text = f"You're in {confirmed_label}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x_center = (frame_disp.shape[1] - tw) // 2
            cv2.putText(frame_disp, text,
                        (x_center, frame_disp.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Live Recognition", frame_disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            picam2.stop()
            cv2.destroyAllWindows()
            popup_root.destroy()
            subprocess.run(["python3", "index.py"]) #redirect to home
            return
        

def log_attendance(dataset_folder):
    file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "folder_name", "Get_time_code") # use this if RPI do not have RCT, gets time from server
    )

    spec = importlib.util.spec_from_file_location("current_time", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    get_time = module.get_time
    now = get_time()

    date_part, time_part = now.split(" ")
    hour = int(time_part.split(":")[0])

    stud_id = dataset_folder #dataset folder is named after the student's registration id

    # check if already logged today
    cursor.execute("""
        SELECT * FROM tbl_attendance
        WHERE stud_Id = ? AND date = ?
    """, (stud_id, date_part))

    already_logged = cursor.fetchone()

    if already_logged:
        return "Already Logged Today"

    # status rule
    if hour < 8:
        status = "On Time"
    else:
        status = "Late" #absent if no remarks upon upload

    cursor.execute("""
        INSERT INTO tbl_attendance (stud_Id, date, remarks)
        VALUES (?, ?, ?)
    """, (stud_id, date_part, status))

    conn.commit()

    return f"Attendance Logged ({status})"
