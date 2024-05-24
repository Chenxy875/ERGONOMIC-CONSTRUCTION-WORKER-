import pyrealsense2 as rs
import numpy as np
import cv2
import cv2
import mediapipe as mp
from angle_calc import angle_calc
import os
from PIL import Image
import matplotlib.pyplot as plt
import time
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
numbers = []
numbers2 = []
# Start streaming
pipeline.start(config)


try:
    while True:
 
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
       
        color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
  
        color_image = np.asanyarray(color_frame.get_data())
 

        # Stack both images horizontally
        images = color_image
 
        # Show images
        #cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        #cv2.imshow('RealSense', images)
 
        frame1=images
        rgb_frame = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb_frame)

        # 绘制姿势关键点及连接线
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(frame1, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            pose1 = []
            for id, lm in enumerate(results.pose_landmarks.landmark):
                x_y_z = []
                h, w, c = frame1.shape
                x_y_z.append(lm.x)
                x_y_z.append(lm.y)
                x_y_z.append(lm.z)
                x_y_z.append(lm.visibility)
                pose1.append(x_y_z)

                cx, cy = int(lm.x * w), int(lm.y * h)
                if id % 2 == 0:
                    cv2.circle(frame1, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                else:
                    cv2.circle(frame1, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            rula, reba = angle_calc(pose1)
            RULA = rula
            REBA = reba
            print(": Rapid Upper Limb Assessment Score=", rula,
                  "Rapid Entire Body Assessment Score=", reba)
            cv2.putText(frame1, f" REBA: {REBA:}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame1, f" RULA: {RULA:}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            #cv2.imwrite(os.path.join(output_dir, f"frame_{len(output_images)}.png"), frame)
            cv2.imshow("frame", frame1)

        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
 
 
finally:
 
    # Stop streaming
    pipeline.stop()