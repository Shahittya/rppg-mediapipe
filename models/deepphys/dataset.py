import numpy as np
import cv2

class DeepPhysDataset:
    def __init__(self,video_path,resize=(72,72)):
        self.video_path = video_path
        self.resize= resize

    def load_video(self):
        cap =cv2.VideoCapture(self.video_path)
        frames = []
        while True:
            ret,frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame,self.resize)
            frame=frame.astype(np.float32)/255.0
            frames.append(frame)
        cap.release()
        return np.array(frames)
    def create_streams(self):
        frames = self.load_video()
        apperance=[]
        motion=[]
        for i in range(len(frames)-1):
            f1=frames[i]
            f2=frames[i+1]
            diff=f2-f1
            apperance.append(f1)
            motion.append(diff)
        return np.array(apperance),np.array(motion)
    

        