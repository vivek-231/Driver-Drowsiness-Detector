import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import threading
import time
import winsound  # For sound alert (on Windows)

# Load Haar Cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize Tkinter window
class DrowsinessDetector:
    def __init__(self, root):
        self.root = root
        self.root.title("Drowsiness Detection System")
        
        # Video capture frame
        self.video_frame = tk.Label(root)
        self.video_frame.pack()
        
        # Start video capture thread
        self.capture = cv2.VideoCapture(0)
        self.is_drowsy = False
        self.update_video()
        
    def update_video(self):
        ret, frame = self.capture.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                
                eyes = eye_cascade.detectMultiScale(roi_gray)
                
                if len(eyes) == 0:  # If no eyes are detected, trigger drowsiness alert
                    if not self.is_drowsy:
                        self.is_drowsy = True
                        threading.Thread(target=self.trigger_alert).start()
                else:
                    self.is_drowsy = False
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
            
            # Convert frame to Tkinter format
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            self.video_frame.config(image=img)
            self.video_frame.image = img
        
        self.root.after(10, self.update_video)
    
    def trigger_alert(self):
        # Display alert message and sound a beep
        messagebox.showwarning("Alert", "Drowsiness Detected!")
        for _ in range(5):
            winsound.Beep(1000, 200)  # Beep sound (Windows only)
            time.sleep(0.2)

    def __del__(self):
        self.capture.release()

# Run the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = DrowsinessDetector(root)
    root.mainloop()
