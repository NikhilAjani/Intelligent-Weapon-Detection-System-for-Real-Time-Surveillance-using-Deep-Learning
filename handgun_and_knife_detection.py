import cv2
import torch
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import pygame
from threading import Thread
from datetime import datetime
from twilio.rest import Client

account_sid = 'twilio_account_sid'
auth_token = 'twilio_auth_token'
twilio_phone_number = 'twilio_phone_number'
user_phone_number = 'your_phone_number'

client = Client(account_sid, auth_token)

gun_model = YOLO(r"path to your gun model")
knife_model = YOLO(r"path to your knife model")

pygame.mixer.init()
pygame.mixer.music.load(r"path to siren.wav")

class SafetyDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Weapon Detection System")
        self.root.geometry("800x500")
        self.root.config(bg="#2E2E2E")

        self.video_capture = None
        self.is_running = False
        self.siren_playing = False

        header = tk.Label(self.root, text="Weapon Detection System",
                          font=("Helvetica", 20, "bold"), bg="#3B3B3B", fg="white")
        header.pack(fill=tk.X, pady=(10, 0))

        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        self.status_label = ttk.Label(main_frame, text="Status: Idle",
                                      font=("Arial", 16), foreground="#FF6347")
        self.status_label.pack(pady=10)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)

        self.start_button = ttk.Button(button_frame, text="Start Detection",
                                       command=self.start_detection, style="TButton")
        self.start_button.grid(row=0, column=0, padx=10)

        self.stop_button = ttk.Button(button_frame, text="Stop Detection",
                                      command=self.stop_detection, style="TButton")
        self.stop_button.grid(row=0, column=1, padx=10)

        self.exit_button = ttk.Button(button_frame, text="Exit",
                                      command=self.exit_application, style="TButton")
        self.exit_button.grid(row=0, column=2, padx=10)

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 14), padding=10)
        style.map("TButton", foreground=[('pressed', 'red'), ('active', 'blue')])

    def start_detection(self):
        """Start detection."""
        if not self.is_running:
            self.is_running = True
            self.video_capture = cv2.VideoCapture(0)
            self.status_label.config(text="Status: Running...", foreground="green")
            Thread(target=self.detect_objects, daemon=True).start()

    def stop_detection(self):
        """Stop detection."""
        self.is_running = False
        if self.video_capture:
            self.video_capture.release()
        self.status_label.config(text="Status: Stopped", foreground="orange")
        if self.siren_playing:
            pygame.mixer.music.stop()
            self.siren_playing = False

    def detect_objects(self):
        """Perform weapon detection."""
        while self.is_running:
            ret, frame = self.video_capture.read()
            if not ret:
                self.status_label.config(text="Status: Camera Error", foreground="red")
                break

            frame = cv2.flip(frame, 1)

            results_gun = gun_model.predict(source=frame, conf=0.5, show=False)
            results_knife = knife_model.predict(source=frame, conf=0.5, show=False)

            annotated_frame_gun = results_gun[0].plot()
            annotated_frame_knife = results_knife[0].plot()

            merged_frame = cv2.addWeighted(annotated_frame_gun, 0.5, annotated_frame_knife, 0.5, 0)

            gun_detected = any(cls in [0, 1] for result in results_gun for cls in result.boxes.cls)
            knife_detected = any("knife" in result.names[int(cls)].lower()
                                for result in results_knife for cls in result.boxes.cls)

            if knife_detected:
                self.status_label.config(text="Status: Knife Detected!", foreground="red")
                self.trigger_alert("Knife")
            elif gun_detected:
                self.status_label.config(text="Status: Gun Detected", foreground="red")
                self.trigger_alert("Gun")
            else:
                self.status_label.config(text="Status: All Safe", foreground="green")

            cv2.imshow("Weapon Detection System", merged_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        self.stop_detection()
        cv2.destroyAllWindows()

    def trigger_alert(self, weapon_type):
        """Trigger siren and send SMS alert."""
        if not self.siren_playing:
            pygame.mixer.music.play(-1)
            self.siren_playing = True

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            message_body = f"Warning! A {weapon_type} has been detected at {current_time}."
            client.messages.create(
                body=message_body,
                from_=twilio_phone_number,
                to=user_phone_number
            )
            print("SMS alert sent successfully!")
        except Exception as e:
            print(f"Failed to send SMS alert: {e}")

    def exit_application(self):
        """Exit the application."""
        self.stop_detection()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SafetyDetectionApp(root)
    root.mainloop()
