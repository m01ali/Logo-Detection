import pandas as pd
import matplotlib.pyplot as plt
from IPython import display
from PIL import Image
from scipy.signal import find_peaks
import numpy as np
import cv2
from scenedetect import detect, AdaptiveDetector

class ShotDetector:
    
    def __init__(self, video_path, stats_path='./video_stats.csv'):
        self.video_path = video_path
        self.stats_path = stats_path
        
    def detect_scenes(self):
        scenes = detect(self.video_path, 
                        AdaptiveDetector(), 
                        stats_file_path=self.stats_path, 
                        show_progress=True)
        df = pd.read_csv(self.stats_path)
        return df
    
    def extract_keyframes(self, df, metric='adaptive_ratio (w=2)', threshold='quantile', quantile=0.98):
        x = df[metric].values
        
        thresh=threshold
        if threshold=='quantile':
            thresh=df[metric].quantile(quantile)
        
        peaks, _ = find_peaks(x, threshold=thresh)
        frames, frame_numbers = self._read_keyframes(peaks)
        timecodes = df.loc[df['Frame Number'].isin(frame_numbers), 'Timecode'].values
        return frames, frame_numbers, timecodes
    
    def process_video(self, **kwargs):
        df = self.detect_scenes()
        return self.extract_keyframes(df, **kwargs)
    
    def _read_keyframes(self, peaks):
        peaks = np.insert(peaks, 0, 0)
        frames = []
        frame_numbers = []
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video")

        amount_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("Total Frames: ", amount_of_frames)


        #Set the video position to the specific frame
        for i in range(len(peaks)-1):
            frame_no = int((peaks[i] + peaks[i+1])/2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            # Read the frame at the current position
            ret, frame = cap.read()

            if ret:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)))
                frame_numbers.append(frame_no)
        return frames, frame_numbers


def main():
    video_path = "C:/Users/Admin/Desktop/milestone 2/videos/2.mp4"
    stats_path= "C:/Users/Admin/Desktop/milestone 2/videos/video_2_stats.csv"
    detector = ShotDetector(video_path=video_path, stats_path=stats_path)
    frames, frame_numbers, timestamps = detector.process_video()
    
    print(len(frames), len(frame_numbers), len(timestamps))
    
if __name__ == "__main__":
    main()