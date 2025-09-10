import cv2
import subprocess
import numpy as np

# Change this to your IR camera name
device_name = "Poly Cam Pro"

# 🔹 Use your exact ffmpeg.exe path (from `where ffmpeg`)
ffmpeg_path = r"C:\Users\MafunoChimpondah\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"

# Desired resolution and fps
width, height, fps = 640, 480, 15

# ffmpeg command
ffmpeg_cmd = [
    ffmpeg_path,
    "-f", "dshow",
    "-i", f"video={device_name}",
    "-pix_fmt", "bgr24",
    "-vcodec", "rawvideo",
    "-an", "-sn",
    "-r", str(fps),             # force frame rate
    "-s", f"{width}x{height}",  # force resolution
    "-f", "rawvideo", "-"
]

# Launch ffmpeg
proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=10**8)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    raw_frame = proc.stdout.read(width * height * 3)
    if not raw_frame:
        break

    # 🔹 Make writable copy
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3)).copy()

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    if len(faces) > 0:
        # Draw rectangles + status
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(frame, "Face Detected ✅", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No Face ❌", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show video
    cv2.imshow("IR Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

proc.kill()
cv2.destroyAllWindows()
