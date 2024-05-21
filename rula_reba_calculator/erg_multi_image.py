import torch
import cv2
import mediapipe as mp
from angle_calc import angle_calc

mpPose = mp.solutions.pose

pose = mpPose.Pose(static_image_mode=True,
                        model_complexity=2,
                        smooth_landmarks=True,
                        enable_segmentation=True,
                        min_detection_confidence=0.6,
                        min_tracking_confidence=0.6)


mpDraw = mp.solutions.drawing_utils

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')


frame = cv2.imread("/home/xhcnegl/桌面/rula_reba_calculator/2.png")

results = model(frame)
boxs = results.pandas().xyxy[0].values



i = 0
for box in boxs:
    if box[-1] == "person":
        minx = int(box[0])
        miny = int(box[1])
        maxx = int(box[2])
        maxy = int(box[3])
        cv2.rectangle(frame, (minx, miny), (maxx, maxy), (0, 255, 0), 2)
        #cv2.putText(frame, box[-1]+str(i),
        #            (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cropped_image = frame[miny:maxy, minx:maxx]

        results = pose.process(cropped_image)
        RULA=[]
        REBA=[] 
        pose1 = []
        if results.pose_landmarks:
            mpDraw.draw_landmarks(cropped_image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            for id, lm in enumerate(results.pose_landmarks.landmark):
                x_y_z = []
                h, w, c = cropped_image.shape
                x_y_z.append(lm.x)
                x_y_z.append(lm.y)
                x_y_z.append(lm.z)
                x_y_z.append(lm.visibility)
                pose1.append(x_y_z)

                #print(lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id % 2 == 0:
                    cv2.circle(cropped_image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                else:
                    cv2.circle(cropped_image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        #print(pose1)
        rula, reba = angle_calc(pose1)
        RULA=rula
        REBA=reba 
                
        print("person", str(i), ": Rapid Upper Limb Assessment Score=", rula, "Rapid Entire Body Assessment Score=", reba)
        if rula != "NULL":

            if float(rula) > 5:
                print("Posture is not ergonomic in upper body")
            if float(reba) > 5:
                print("Posture is not ergonomic in body")
    i+=1

cv2.putText(frame, f" RULA: {RULA:}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
cv2.putText(frame, f" REBA: {REBA:}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)
#cv2.imshow("cropped", cropped_image)
cv2.imshow("frame", frame)

key = cv2.waitKey(0)
if key == ord('q'): # 如果按下的是q键
  cv2.destroyAllWindows() # 关闭所有的窗口


  