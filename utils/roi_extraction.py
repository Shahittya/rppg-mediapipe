import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh=mp.solutions.face_mesh

class ROIExtractor:
    def __init__(self):
        self.face_mesh=mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1
        )
    def get_rois(self, frame):
        h,w,_=frame.shape
        rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return None
        face_landmarks=results.multi_face_landmarks[0]

        def get_region(indices):
            pts=[]
            for idx in indices:
                lm=face_landmarks.landmark[idx]
                x,y=int(lm.x*w),int(lm.y*h)
                pts.append((x,y))
            pts = np.array(pts, dtype=np.int32)
            center = np.mean(pts, axis=0)
            angles = np.arctan2(pts[:,1] - center[1], pts[:,0] - center[0])
            pts = pts[np.argsort(angles)]
            return pts
        # Define regions
        forehead_idx = [10, 109, 67, 103,54,        
                        338, 297, 332, 284, 251]
        
        left_cheek_idx = [50, 101, 118, 119, 120,
                          123, 147, 187, 205,
                          207, 213,216,192]
        right_cheek_idx = [280, 330, 347, 348, 349,
                           352, 376, 411, 425,427, 433,436,416]

        forehead_idx=get_region(forehead_idx)
        left_cheek_idx=get_region(left_cheek_idx) 
        right_cheek_idx=get_region(right_cheek_idx)
        return {
            "forehead": forehead_idx,
            "left_cheek_idx": left_cheek_idx,
            "right_cheek_idx": right_cheek_idx
        }     
    def extract_roi(self, frame, polygon):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        roi = cv2.bitwise_and(frame, frame, mask=mask)
        # Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        #Relaxed skin range
        lower = np.array([0, 10, 40])
        upper = np.array([35, 200, 255])
        skin_mask = cv2.inRange(hsv, lower, upper)
        #Fill gaps
        kernel = np.ones((5,5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        roi = cv2.bitwise_and(roi, roi, mask=skin_mask)
        #Smooth
        roi = cv2.GaussianBlur(roi, (5,5), 0)
        return roi