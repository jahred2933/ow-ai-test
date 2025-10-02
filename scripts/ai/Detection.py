import mss
import cv2
import numpy as np
import time
import pandas as pd
from ultralytics import YOLO
import keyboard
import math
import pydirectinput

# Kalman Filter class
class KalmanFilter:
    def __init__(self, process_noise=1e-5, measurement_noise=1e-1):
        self.x = np.array([[0], [0]])  # State vector (position and velocity)
        self.P = np.eye(2) * 1000  # Covariance matrix
        self.F = np.array([[1, 1], [0, 1]])  # Transition matrix
        self.H = np.array([[1, 0]])  # Measurement matrix
        self.R = np.array([[measurement_noise]])  # Measurement noise
        self.Q = np.array([[process_noise, 0], [0, process_noise]])  # Process noise
        self.I = np.eye(2)  # Identity matrix

    def predict(self):
        """Predict the next state"""
        self.x = np.dot(self.F, self.x)  # Predicted state
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # Predicted covariance
        return self.x

    def update(self, z):
        """Update the state with new measurement"""
        y = z - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # Residual covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        self.x = self.x + np.dot(K, y)  # Update state estimate
        self.P = np.dot(self.I - np.dot(K, self.H), self.P)  # Update covariance
        return self.x

class Detection:
    def __init__(self):
        screen_width, screen_height = 2560, 1440  
        print(f"Detected screen resolution: {screen_width}x{screen_height}")

        MONITOR_WIDTH = 2560
        MONITOR_HEIGHT = 1440
        MONITOR_SCALE = 1  
        region = (int(MONITOR_WIDTH / 2 - MONITOR_WIDTH / MONITOR_SCALE / 2),
                  int(MONITOR_HEIGHT / 2 - MONITOR_HEIGHT / MONITOR_SCALE / 2),
                  int(MONITOR_WIDTH / 2 + MONITOR_WIDTH / MONITOR_SCALE / 2),
                  int(MONITOR_HEIGHT / 2 + MONITOR_HEIGHT / MONITOR_SCALE / 2))
        self.screenshotCenter = [int((region[2] - region[0]) / 2), int((region[3] - region[1]) / 2)]

        # Load YOLO model
        self.model = YOLO('../models/v2.pt')  
        self.model.to("cuda")
        self.model.conf = 0.80  
        self.model.maxdet = 10  
        self.model.apm = True  

        # Aim & Trigger Settings
        self.settings = {
            "toggleKeyTriggerbot": "f2",
            "cooldown": 0.005,  
            "detect": [0, 1],  
            "triggerDelay": 0,
            "holdTrigger": True,
            "aim_speed": 420,  # Base aim speed
            "max_speed": 520,  # Maximum speed for movement
            "min_speed": 1,   # Minimum speed for fine adjustments
            "prefire_time": 0.0,  # Time to prefire (seconds)
        }

        self.triggerbot = False
        self.lastClick = 0
        self.kf = KalmanFilter()  # Initialize Kalman filter
        self.previous_position = [self.screenshotCenter[0], self.screenshotCenter[1]]

        self.run_detection(region)

    def run_detection(self, region):
        with mss.mss() as stc:
            while True:
                currentTime = time.time()
                screenshot = np.array(stc.grab(region))
                screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGRA2BGR)

                # YOLO Inference
                frame = self.model.predict(screenshot, save=False, classes=self.settings["detect"], verbose=False, device=0, half=True)
                positionsFrame = pd.DataFrame(frame[0].cpu().numpy().boxes.data, columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])

                # Find closest enemy head
                closestPartDistance = float('inf')
                closestPart = -1
                for i, row in positionsFrame.iterrows():
                    try:
                        xmin, ymin, xmax, ymax, confidence, category = row.astype('int')
                        target_centerX = (xmax + xmin) / 2
                        target_centerY = (ymax + ymin) / 2
                        distance = math.dist([target_centerX, target_centerY], self.screenshotCenter)

                        if distance < closestPartDistance:
                            closestPartDistance = distance
                            closestPart = i

                        cv2.rectangle(screenshot, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
                    except Exception as e:
                        print(f"Error in position extraction: {e}", end="")

                # Handle toggles
                self.handle_key_toggles(currentTime)

                if self.triggerbot and closestPart != -1:  # ✅ Aiming only when triggerbot is on
                    xmin = positionsFrame.iloc[closestPart, 0]
                    ymin = positionsFrame.iloc[closestPart, 1]
                    xmax = positionsFrame.iloc[closestPart, 2]
                    ymax = positionsFrame.iloc[closestPart, 3]
                    # Use the center of the bounding box as the target
                    target_centerX = (xmax + xmin) / 2
                    target_centerY = (ymax + ymin) / 2

                    # Apply Kalman Filter to smooth the detection
                    self.kf.update(np.array([[target_centerX]]))  # Update with new center position
                    predicted_center = self.kf.predict()  # Get predicted position

                    # Calculate prefire position: Predicted position + prefire time * velocity
                    velocity = [target_centerX - self.previous_position[0], target_centerY - self.previous_position[1]]
                    prefireX = target_centerX + self.settings["prefire_time"] * velocity[0]
                    prefireY = target_centerY + self.settings["prefire_time"] * velocity[1]

                    # Update previous position for future velocity calculation
                    self.previous_position = [target_centerX, target_centerY]

                    # Smoother mouse movement, independent of YOLO frame rate
                    self.smooth_move_to_target(prefireX, prefireY)

                    # Triggerbot Logic
                    inRange = self.screenshotCenter[0] in range(int(xmin), int(xmax)) and self.screenshotCenter[1] in range(int(ymin), int(ymax))
                    if inRange:
                        if currentTime - self.lastClick > self.settings["cooldown"]:
                            time.sleep(self.settings["triggerDelay"])
                            if not keyboard.is_pressed('left'):  # Only trigger if you're not holding the left mouse
                                pydirectinput.mouseDown() if self.settings["holdTrigger"] else pydirectinput.click()
                            self.lastClick = currentTime
                    else:
                        pydirectinput.mouseUp()  # Release if head no longer detected
                else:
                    pydirectinput.mouseUp()  # ✅ Release mouse if triggerbot is OFF or no target

                # Display Status
                self.display_status(screenshot)

                if cv2.waitKey(1) == ord('l'):
                    cv2.destroyAllWindows()
                    break

    def handle_key_toggles(self, currentTime):
        if keyboard.is_pressed(self.settings["toggleKeyTriggerbot"]) and currentTime - self.lastClick > 0.2:
            self.triggerbot = not self.triggerbot
            self.lastClick = currentTime
            print(f"Triggerbot: {self.triggerbot}")

    def display_status(self, screenshot):
        cv2.rectangle(screenshot, (0, 0), (20, 20), (0, 255, 0) if self.triggerbot else (0, 0, 255), -1)
        cv2.putText(screenshot, "Triggerbot", (50, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.imshow("frame", screenshot)

    def smooth_move_to_target(self, targetX, targetY):
        """Move the mouse smoothly towards the target with a high-frequency update"""
        # Current mouse position
        currentX, currentY = self.screenshotCenter
        deltaX = targetX - currentX
        deltaY = targetY - currentY

        # Calculate distance
        distance = math.sqrt(deltaX ** 2 + deltaY ** 2)

        if distance > 1:  # Only move if the target is far enough
            # Calculate speed (variable based on the distance to target)
            speed = self.settings["aim_speed"] * (distance / 1000)  # Smooth speed scaling
            speed = min(max(speed, self.settings["min_speed"]), self.settings["max_speed"])  # Clamp speed to a range

            # Calculate step (linear interpolation)
            moveX = int(deltaX * speed / distance)
            moveY = int(deltaY * speed / distance)

            # Apply smooth movement
            pydirectinput.moveRel(moveX, moveY, relative=True)

if __name__ == "__main__":
    Detection()




























