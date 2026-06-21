import cv2
import os
import numpy as np
import threading
from playsound import playsound   # pip install playsound==1.2.2
from insightface.app import FaceAnalysis   # pip install insightface

# Path to directory containing subfolders of known people with their photos
KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 1.2   # Smaller = stricter match
FRAME_THICKNESS = 3
FONT_THICKNESS = 2

# Path to alert sound
ALERT_SOUND = r"C:\Users\LEKHRAJ\Downloads\police-siren-397963.mp3"

# ✅ Known persons descriptions
person_descriptions = {
    "Lekhraj Singh": "Btech  student roll number 221168",
    "Sujal Saini": "Btech student roll number 221123",
    "Roshan": "Btech student roll nummber 221161",
    # Add more people here matching folder names
}

# 🔊 Thread-safe sound function
def play_alert():
    try:
        playsound(ALERT_SOUND)
    except Exception as e:
        print("Error playing sound:", e)

# Load InsightFace model
print("Loading InsightFace model...")
app = FaceAnalysis(providers=['CPUExecutionProvider'])  # change to GPU if available
app.prepare(ctx_id=0, det_size=(640, 640))

known_faces = []
known_names = []

# Load known faces
print("Loading known faces...")
for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        img_path = os.path.join(KNOWN_FACES_DIR, name, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = app.get(img)
        if faces:
            known_faces.append(faces[0].normed_embedding)
            known_names.append(name)

print(f"Loaded {len(known_faces)} known faces.")

# Start webcam
video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        embedding = face.normed_embedding
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        match = "Unknown"
        description = ""
        color = (0, 0, 255)  # red for unknown

        # Compare face with known faces
        if known_faces:
            distances = np.linalg.norm(np.array(known_faces) - embedding, axis=1)
            best_match_index = np.argmin(distances)
            if distances[best_match_index] < TOLERANCE:
                match = known_names[best_match_index]
                color = (0, 255, 0)  # green for known
                # ✅ Known person description
                description = person_descriptions.get(match, "No info available")
            else:
                match = "Unknown"
                color = (0, 0, 255)
                description = "Not Found"
        else:
            description = "Not Found"

        # ==========================
        # FACE-ONLY ACTIVE EFFECT
        # ==========================
        overlay = frame.copy()
        overlay[:] = (0, 0, 0)   # dark background
        alpha = 0.6              # background transparency

        center = (x1 + (x2 - x1)//2, y1 + (y2 - y1)//2)
        axes = ((x2 - x1)//2, (y2 - y1)//2)

        mask = np.zeros_like(frame, dtype=np.uint8)
        cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)

        darkened = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        frame = np.where(mask == (255, 255, 255), frame, darkened)

        # ==========================
        # DRAW ELLIPSE + GRID
        # ==========================
        cv2.ellipse(frame, center, axes, 0, 0, 360, color, FRAME_THICKNESS)

        num_ellipses = 2  # concentric ellipses
        for i in range(1, num_ellipses + 1):
            cv2.ellipse(frame, center,
                        (axes[0] * i // (num_ellipses + 1),
                         axes[1] * i // (num_ellipses + 1)),
                        0, 0, 360, color, 1)

        num_radial = 4  # radial lines like grid
        for i in range(num_radial):
            angle = i * (180 // (num_radial + 1))
            x = int(center[0] + axes[0] * np.cos(np.radians(angle)))
            y = int(center[1] + axes[1] * np.sin(np.radians(angle)))
            cv2.line(frame, center, (x, y), color, 1)

        # ==========================
        # SHOW NAME + DESCRIPTION
        # ==========================
        cv2.putText(frame, match, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, FONT_THICKNESS)

        if description:
            cv2.putText(frame, description, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

        # 🚨 Play sound if unknown
        if match == "Unknown":
            threading.Thread(target=play_alert, daemon=True).start()

    # Show frame
    cv2.imshow("Face Recognition (Active Face Only + Grid)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
