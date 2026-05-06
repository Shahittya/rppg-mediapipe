import cv2
import torch
import numpy as np
from utils.roi_extraction import ROIExtractor
from utils.signal_extraction import SignalExtractor
import matplotlib.pyplot as plt
from utils.signal_processing import process_signal
from utils.chrom import chrom_method
from utils.heart_rate import calculate_heart_rate
from models.deepphys.dataset import DeepPhysDataset
from models.deepphys.model import DeepPhysModel
from models.deepphys.dataset_loader import RPPGDataset
video_path = "data/videos/subject3/03fdb810e50b4aa58edbccc6012c6710_1.mp4"

cap = cv2.VideoCapture(video_path)
roi_extractor = ROIExtractor()
signal_extractor = SignalExtractor()

frame_count = 0 

# Create resizable window 
cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
fig,ax=plt.subplots(figsize=(8,4))
while True:
    ret, frame = cap.read()
    if not ret:
        print("Video finished")
        break
    frame_count += 1
    rois = roi_extractor.get_rois(frame)

    if rois:
        for name, polygon in rois.items():
            polygon = cv2.convexHull(polygon)
            roi = roi_extractor.extract_roi(frame, polygon)
            # Extract RGB signal
            if "forehead" in name:
                name = "forehead"
            elif "left_cheek" in name:
                name = "left_cheek" 
            elif "right_cheek" in name:
                name = "right_cheek"
            else:
                continue
            signal_extractor.extract_rgb(roi, name)
            # Draw ROI boundary
            cv2.polylines(frame, [polygon], True, (0, 255, 0), 2)
            # Show each ROI window
            cv2.imshow(name, roi)
            #Graph Update
            
    # Show main frame
    cv2.imshow("Frame", frame)
    # show progress every 100 frames
    if frame_count % 100 == 0:
        print(f"Processed frames: {frame_count}")
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:  # q OR ESC
        print("Exiting loop...")
        break
#r, g, b = signal_extractor.get_signals()
# signals=signal_extractor.get_signals()
# def normalize(signal):
#     signal = np.array(signal)
#     std=np.std(signal)
#     if std == 0:
#         return signal
#     return (signal - np.mean(signal)) / std
# #left cheek signal processing
# l_r=process_signal(signals["left_cheek"]["r"])
# l_g=process_signal(signals["left_cheek"]["g"])
# l_b=process_signal(signals["left_cheek"]["b"])
# #left cheek normalization
# l_r=normalize(l_r)
# l_g=normalize(l_g)
# l_b=normalize(l_b)
# #right cheek signal processing
# r_r=process_signal(signals["right_cheek"]["r"])
# r_g=process_signal(signals["right_cheek"]["g"])
# r_b=process_signal(signals["right_cheek"]["b"])
# #right cheek normalization
# r_r=normalize(r_r)
# r_g=normalize(r_g)
# r_b=normalize(r_b)
# #forehead signal processing
# f_r=process_signal(signals["forehead"]["r"])
# f_g=process_signal(signals["forehead"]["g"])
# f_b=process_signal(signals["forehead"]["b"])
# #forehead normalization
# f_r=normalize(f_r)
# f_g=normalize(f_g)
# f_b=normalize(f_b)
# l_chrom=chrom_method(l_r,l_g,l_b)
# r_chrom=chrom_method(r_r,r_g,r_b)
# f_chrom=chrom_method(f_r,f_g,f_b)
# fps=cap.get(cv2.CAP_PROP_FPS)
# f_bpm,f_freqs,f_fft=calculate_heart_rate(f_chrom,fps)
# l_bpm,l_freqs,l_fft=calculate_heart_rate(l_chrom,fps)
# r_bpm,r_freqs,r_fft=calculate_heart_rate(r_chrom,fps)

# print("\n===== HEART RATE =====")
# print(f"Forehead BPM: {f_bpm:.2f}")
# print(f"Left Cheek BPM: {l_bpm:.2f}")
# print(f"Right Cheek BPM: {r_bpm:.2f}")
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("FPS not detected, using default 30")
    fps = 30
# Release resources
cap.release()
cv2.destroyAllWindows()
# dataset=DeepPhysDataset(video_path)
# appearance,motion = dataset.create_streams()
# appearance=torch.tensor(appearance,dtype=torch.float32)
# motion=torch.tensor(motion,dtype=torch.float32)
# model=DeepPhysModel()
# output=model(appearance,motion)
# deep_signal=output.detach().numpy()
# deep_signal=(deep_signal - np.mean(deep_signal))/np.std(deep_signal)
# bpm,freqs,fft_vals=calculate_heart_rate(deep_signal,fps)
# print(f"DeepPhys BPM: {bpm:.2f}")
# plt.figure(figsize=(8,4))
# plt.plot(freqs, fft_vals)
# plt.title("DeepPhys Frequency Spectrum")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.show()
# plt.plot(f_g,label="Forehead(G) raw", alpha=0.3)
# plt.plot(l_g,label="Left Cheek(G) raw", alpha=0.3)
# plt.plot(r_g,label="Right Cheek(G) raw", alpha=0.3)
# plt.plot(f_chrom,label="Forehead Chrom", linewidth=2)
# plt.plot(l_chrom,label="Left Cheek Chrom", linewidth=2)
# plt.plot(r_chrom,label="Right Cheek Chrom", linewidth=2)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.title("rPPG Signals")=
# plt.figure(figsize=(8,4))
# plt.plot(f_freqs, f_fft,label="FFT Spectrum")
# plt.title("Frequency Spectrum (Forehead)")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.legend()
# plt.figure(figsize=(8,4))
# plt.plot(l_freqs, l_fft,label="FFT Spectrum")
# plt.title("Frequency Spectrum (Left Cheek)")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.legend()
# plt.figure(figsize=(8,4))
# plt.plot(r_freqs, r_fft,label="FFT Spectrum")
# plt.title("Frequency Spectrum (Right Cheek)")
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.legend()
# plt.show()
