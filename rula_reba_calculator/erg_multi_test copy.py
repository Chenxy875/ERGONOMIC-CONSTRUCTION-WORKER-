import torch
import cv2
import mediapipe as mp
from angle_calc import angle_calc
import time
import matplotlib.pyplot as plt
numbers=[]
mpPose = mp.solutions.pose
pose = mpPose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                      min_tracking_confidence=0.7)

'''pose = mpPose.Pose(static_image_mode=False,
                        model_complexity=2,
                        enable_segmentation=False,
                        min_detection_confidence=0.2,
                        min_tracking_confidence=0.1)'''

mpDraw = mp.solutions.drawing_utils

model = torch.hub.load('ultralytics/yolov5', 'yolov5l')


cap = cv2.VideoCapture('/home/xhcnegl/桌面/data/5.mp4')

frame_count = 0
total_fps = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_no = frame_count*10
    cap.set(1, frame_no)

    start_time = time.time()
    RULA="NULL"
    REBA="NULL"
    try:
        results = model(frame)
        boxs = results.pandas().xyxy[0].values

        i = 0
        for box in boxs:
            if box[-1] == "person":
                minx = int(box[0])-10
                miny = int(box[1])-10
                maxx = int(box[2])+10
                maxy = int(box[3])+10
                cv2.rectangle(frame, (minx, miny), (maxx, maxy), (0, 255, 0), 2)
                #cv2.putText(frame, box[-1] + str(i),
                #           (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                cropped_image = frame[miny:maxy, minx:maxx]

                results = pose.process(cropped_image)
                pose1 = []
                if results.pose_landmarks:
                    mpDraw.draw_landmarks(cropped_image, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                    for id, lm in enumerate(results.pose_landmarks.landmark):
                        x_y_z = []
                        h, w, c = frame.shape
                        x_y_z.append(lm.x)
                        x_y_z.append(lm.y)
                        x_y_z.append(lm.z)
                        x_y_z.append(lm.visibility)
                        pose1.append(x_y_z)
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if id % 2 == 0:
                            cv2.circle(cropped_image, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
                        else:
                            cv2.circle(cropped_image, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                rula, reba = angle_calc(pose1)
                RULA=rula
                REBA=reba 
                numbers.append(REBA)
                print(numbers)

                print("person", str(i), ": Rapid Upper Limb Assessment Score=", rula,
                      "Rapid Entire Body Assessment Score=", reba)
                if float(rula) > 5:
                    print("Posture is not ergonomic in upper body")
                if float(reba) > 5:
                    print("Posture is not ergonomic in body")
            i += 1

    except:
        print("An error occured.")

    end_time = time.time()
    fps = 1 / (end_time - start_time)

    frame_count += 1
    total_fps += fps

    cv2.putText(frame, f" RULA: {RULA:}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)
    cv2.putText(frame, f" REBA: {REBA:}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 2)

    cv2.imshow("frame", frame)


    if cv2.waitKey(0) & 0xFF == ord('q'):
        break



avg_fps = total_fps/frame_count
print(f"Average fps: {avg_fps:.3f}")

cap.release()
cv2.destroyAllWindows()


data = [int(x) for x in numbers]

# 创建x轴和y轴数据
x = range(1, len(data) + 1)
y = data

# 绘制线图
plt.plot(x, y, marker='o')

# 添加标题和轴标签
plt.title('Line Chart')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图形
plt.savefig('line_chart.png')
