"""
Step 3: Live spit detection using webcam.
Run: python 3_live_detect.py
     python 3_live_detect.py --arduino COM3          (Windows)
     python 3_live_detect.py --arduino /dev/ttyUSB0  (Linux/Mac)
     python 3_live_detect.py --weights path/to/best.pt
     python 3_live_detect.py --conf 0.5
     python 3_live_detect.py --camera 1              (external webcam)
"""

import argparse
import os
import sys
import time
import threading
import cv2
import numpy as np
import torch

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS    = "runs/train/spit_detector/weights/best.pt"
CONF_THRESHOLD     = 0.45
IOU_THRESHOLD      = 0.45
IMG_SIZE           = 640
CAMERA_INDEX       = 0
ALERT_DURATION_SEC = 3.0     # How long to show alert after detection
ALERT_COOLDOWN_SEC = 2.0     # Minimum gap between Arduino sends
# ──────────────────────────────────────────────────────────────────────────────


# ─── 7-SEGMENT FONT (on-screen simulation) ────────────────────────────────────
# Each character is a 3×5 pixel bitmap (list of rows, MSB=left)
SEG_FONT = {
    'D': [[1,1,0],[1,0,1],[1,0,1],[1,0,1],[1,1,0]],
    'O': [[0,1,0],[1,0,1],[1,0,1],[1,0,1],[0,1,0]],
    'N': [[1,0,1],[1,1,1],[1,1,1],[1,0,1],[1,0,1]],
    'T': [[1,1,1],[0,1,0],[0,1,0],[0,1,0],[0,1,0]],
    'S': [[0,1,1],[1,0,0],[0,1,0],[0,0,1],[1,1,0]],
    'P': [[1,1,0],[1,0,1],[1,1,0],[1,0,0],[1,0,0]],
    'I': [[1,1,1],[0,1,0],[0,1,0],[0,1,0],[1,1,1]],
    ' ': [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]],
    '!': [[0,1,0],[0,1,0],[0,1,0],[0,0,0],[0,1,0]],
}

def draw_seg_char(frame, char, x, y, scale=8, color=(0, 255, 0), gap=2):
    """Draw a single 7-segment-style character on the frame."""
    bitmap = SEG_FONT.get(char.upper(), SEG_FONT[' '])
    for row_idx, row in enumerate(bitmap):
        for col_idx, pixel in enumerate(row):
            if pixel:
                px = x + col_idx * (scale + gap)
                py = y + row_idx * (scale + gap)
                cv2.rectangle(frame,
                              (px, py),
                              (px + scale, py + scale),
                              color, -1)

def draw_seg_text(frame, text, x, y, scale=8, color=(0, 255, 0), char_gap=4):
    """Draw a full string in 7-segment style."""
    char_width = 3 * (scale + 2) + char_gap
    for i, ch in enumerate(text):
        draw_seg_char(frame, ch, x + i * char_width, y, scale, color)


# ─── ARDUINO SERIAL ───────────────────────────────────────────────────────────
class ArduinoSerial:
    def __init__(self, port, baud=9600):
        self.port = port
        self.ser = None
        self._last_send = 0
        try:
            import serial
            self.ser = serial.Serial(port, baud, timeout=1)
            time.sleep(2)  # Let Arduino reset
            print(f"✅ Arduino connected on {port}")
        except ImportError:
            print("WARNING: pyserial not installed. Run: pip install pyserial")
        except Exception as e:
            print(f"WARNING: Could not connect to Arduino on {port}: {e}")

    def send_alert(self):
        """Send alert command to Arduino (rate-limited)."""
        if self.ser and self.ser.is_open:
            now = time.time()
            if now - self._last_send >= ALERT_COOLDOWN_SEC:
                self._last_send = now
                try:
                    self.ser.write(b'SPIT\n')
                    print("[Arduino] Sent: SPIT")
                except Exception as e:
                    print(f"[Arduino] Send error: {e}")

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.write(b'CLEAR\n')
            time.sleep(0.5)
            self.ser.close()


# ─── ALERT OVERLAY ────────────────────────────────────────────────────────────
class AlertOverlay:
    def __init__(self):
        self._active_until = 0
        self._lock = threading.Lock()
        self._scroll_offset = 0
        self._message = "DO NOT SPIT! "
        self._last_scroll = time.time()

    def trigger(self):
        with self._lock:
            self._active_until = time.time() + ALERT_DURATION_SEC

    @property
    def active(self):
        with self._lock:
            return time.time() < self._active_until

    def draw(self, frame):
        if not self.active:
            return frame

        h, w = frame.shape[:2]
        now = time.time()

        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 180), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Red border flash
        alpha = abs(np.sin(now * 4))  # 0–1 pulsing
        border_color = (0, 0, int(255 * alpha + 100))
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, 8)

        # Scrolling 7-segment text at bottom
        if now - self._last_scroll > 0.08:
            self._scroll_offset = (self._scroll_offset + 1) % len(self._message)
            self._last_scroll = now

        scrolled = (self._message * 3)[self._scroll_offset: self._scroll_offset + 14]

        seg_scale = 10
        char_w = 3 * (seg_scale + 2) + 4
        text_w = len(scrolled) * char_w
        start_x = (w - text_w) // 2
        draw_seg_text(frame, scrolled, start_x, h - 130,
                      scale=seg_scale, color=(0, 255, 0))

        # Bold text warning
        font = cv2.FONT_HERSHEY_DUPLEX
        msg = "⚠  DO NOT SPIT  ⚠"
        ts, _ = cv2.getTextSize(msg, font, 1.4, 3)
        tx = (w - ts[0]) // 2
        cv2.putText(frame, msg, (tx + 2, h - 22), font, 1.4, (0, 0, 0), 5)
        cv2.putText(frame, msg, (tx, h - 24), font, 1.4, (0, 80, 255), 3)

        return frame


# ─── DETECTION LOOP ───────────────────────────────────────────────────────────
def load_model(weights_path):
    print(f"Loading model from: {weights_path}")
    if not os.path.exists(weights_path):
        print(f"ERROR: Weights not found at '{weights_path}'")
        print("Run 2_train.py first, or specify --weights path/to/best.pt")
        sys.exit(1)

    # Use YOLOv5 torch hub loader
    sys.path.insert(0, "yolov5")
    model = torch.hub.load(
        "yolov5",
        "custom",
        path=weights_path,
        source="local",
        verbose=False,
    )
    model.conf = CONF_THRESHOLD
    model.iou  = IOU_THRESHOLD
    model.imgsz = IMG_SIZE
    print("✅ Model loaded!")
    return model


def run_detection(args):
    model   = load_model(args.weights)
    cap     = cv2.VideoCapture(args.camera)
    alert   = AlertOverlay()
    arduino = ArduinoSerial(args.arduino) if args.arduino else None

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera index {args.camera}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n🎥 Live detection started. Press Q to quit.\n")

    fps_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        frame_count += 1

        # ── Run inference ──
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(rgb, size=IMG_SIZE)

        spit_detected = False
        detections = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,class]

        for *box, conf, cls in detections:
            x1, y1, x2, y2 = map(int, box)
            label = f"SPIT {conf:.2f}"
            spit_detected = True

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            bg_x2 = x1 + cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] + 8
            cv2.rectangle(frame, (x1, y1 - 28), (bg_x2, y1), (0, 0, 255), -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if spit_detected:
            alert.trigger()
            if arduino:
                arduino.send_alert()

        # ── Draw alert overlay ──
        frame = alert.draw(frame)

        # ── FPS counter ──
        elapsed = time.time() - fps_time
        if elapsed > 0:
            fps = frame_count / elapsed
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # ── Status indicator ──
        status_text = "DETECTING..." if not alert.active else "⚠ SPIT DETECTED"
        status_color = (0, 200, 0) if not alert.active else (0, 80, 255)
        cv2.putText(frame, status_text, (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)

        cv2.imshow("Spit Detection — Press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    if arduino:
        arduino.close()
    print("Detection stopped.")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Live Spit Detection")
    parser.add_argument("--weights", type=str,
                        default=DEFAULT_WEIGHTS,
                        help="Path to trained YOLOv5 weights (.pt)")
    parser.add_argument("--conf", type=float,
                        default=CONF_THRESHOLD,
                        help="Confidence threshold (0–1)")
    parser.add_argument("--camera", type=int,
                        default=CAMERA_INDEX,
                        help="Camera index (0=default laptop camera)")
    parser.add_argument("--arduino", type=str,
                        default=None,
                        help="Serial port for Arduino 7-segment display (e.g. COM3 or /dev/ttyUSB0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CONF_THRESHOLD = args.conf
    run_detection(args)
