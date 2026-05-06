import cv2
import torch
import numpy as np

from utils.roi_extraction import ROIExtractor
from utils.signal_extraction import SignalExtractor
from utils.signal_processing import process_signal
from utils.chrom import chrom_method
from utils.heart_rate import calculate_heart_rate
from utils.fusion import normalize, select_best
from models.deepphys.model import DeepPhysLSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = DeepPhysLSTM().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
print(f"Model loaded on {device}")

video_path = "data/videos/subject3/03fdb810e50b4aa58edbccc6012c6710_1.mp4"
cap        = cv2.VideoCapture(video_path)
fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f"FPS: {fps}")

# Face detector for cropping — same as dataset_loader now
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

roi_extractor    = ROIExtractor()
signal_extractor = SignalExtractor()

WINDOW_SIZE = 150
STEP_SIZE   = 75
CNN_SIZE    = (72, 72)

cnn_frames, cnn_motions = [], []
prev_frame   = None
final_bpms   = []
prev_bpm     = None
frame_count  = 0

cv2.namedWindow("rPPG", cv2.WINDOW_NORMAL)

while True:
    ret, raw = cap.read()
    if not ret:
        break
    frame_count += 1

    display = cv2.resize(raw, (320, 240))

    # ROI on display frame
    rois = roi_extractor.get_rois(display)
    if rois:
        for name, poly in rois.items():
            hull = cv2.convexHull(poly)
            cv2.polylines(display, [hull], True, (0,255,0), 2)
            roi = roi_extractor.extract_roi(display, hull)
            if   "forehead" in name: region = "forehead"
            elif "left"     in name: region = "left_cheek"
            elif "right"    in name: region = "right_cheek"
            else: continue
            signal_extractor.extract_rgb(roi, region)

    cv2.imshow("rPPG", display)
    if frame_count % 30 == 0:
        print(f"Frames: {frame_count}")

    # CNN input: face-cropped, 72x72, normalised [0,1]
    gray  = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(50,50))
    if len(faces) > 0:
        x, y, w, h = faces[0]
        margin = int(0.1 * max(w,h))
        x1,y1 = max(0,x-margin), max(0,y-margin)
        x2,y2 = min(raw.shape[1],x+w+margin), min(raw.shape[0],y+h+margin)
        cnn_raw = raw[y1:y2, x1:x2]
    else:
        cnn_raw = raw   # fallback if no face detected

    cnn_frame = cv2.resize(cnn_raw, CNN_SIZE).astype(np.float32) / 255.0

    # Normalised motion: DeepPhys paper formula (C(t+1)-C(t))/(C(t+1)+C(t))
    if prev_frame is None:
        motion = np.zeros((*CNN_SIZE, 3), dtype=np.float32)
    else:
        denom  = cnn_frame + prev_frame + 1e-6
        motion = (cnn_frame - prev_frame) / denom

    prev_frame = cnn_frame
    cnn_frames.append(cnn_frame)
    cnn_motions.append(motion)

    if len(cnn_frames) == WINDOW_SIZE:
        print("\n--- Window ---")
        app_t = torch.from_numpy(np.array(cnn_frames,  np.float32)).to(device)
        mot_t = torch.from_numpy(np.array(cnn_motions, np.float32)).to(device)

        with torch.no_grad():
            deep = model(app_t, mot_t).cpu().numpy()
        deep = normalize(deep)

        sigs = signal_extractor.get_signals()
        if len(sigs["forehead"]["r"]) >= 30:
            r = process_signal(sigs["forehead"]["r"])
            g = process_signal(sigs["forehead"]["g"])
            b = process_signal(sigs["forehead"]["b"])
            chrom = normalize(chrom_method(r, g, b, fs=fps))
            n     = min(len(chrom), len(deep))
            sel   = select_best(chrom[:n], deep[:n], fs=fps)
            bpm, _, _ = calculate_heart_rate(sel, fps, prev_bpm)
            print(f"BPM: {bpm:.1f} | Prev: {prev_bpm}")
            prev_bpm = bpm
            final_bpms.append(bpm)
        else:
            print("Skipping — no ROI")

        cnn_frames       = cnn_frames[STEP_SIZE:]
        cnn_motions      = cnn_motions[STEP_SIZE:]
        signal_extractor = SignalExtractor()

    if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
        break

cap.release()
cv2.destroyAllWindows()

if final_bpms:
    avg = sum(final_bpms) / len(final_bpms)
    gt  = 81.3
    print(f"\n===== RESULT =====")
    print(f"Windows : {[round(b,1) for b in final_bpms]}")
    print(f"Avg BPM : {avg:.1f}")
    print(f"GT mean : {gt} BPM")
    print(f"MAE     : {abs(avg-gt):.1f} BPM")
else:
    print("No BPM computed.")
