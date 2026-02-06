import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
mp_drawing = mp.solutions.drawing_utils 

def show_landmarks(image, pose):
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imageRGB.flags.writeable = False 
    results = pose.process(imageRGB)
    imageRGB.flags.writeable = True
    height, width, _ = image.shape
    landmarks = []
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,
                                  connections=mp_pose.POSE_CONNECTIONS)
        for landmark in results.pose_landmarks.landmark:
            landmarks.append(np.array([int(landmark.x * width), 
                                       int(landmark.y * height)
                                    #    int(landmark.z * width)
                                       ]))
        # landmarks = np.array(landmarks)
    else:
        landmarks = np.array([])
    return output_image, landmarks

def calculate_angle(landmark1, landmark2, landmark3):

    a = landmark1[:2]
    b = landmark2[:2]
    c = landmark3[:2]

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    return angle

def detectPose(output_image, landmarks):

    # 兩嘴角中心點:嘴角左[9]、嘴角右[10]
    mouth_center = (landmarks[mp_pose.PoseLandmark.MOUTH_LEFT.value] + landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT.value]) / 2
    cv2.circle(output_image, (int(mouth_center[0]), int(mouth_center[1])), 8, (0, 255, 0), -1)

    # 左手肘角度:左肩[11]、左肘[13]、左腕[15]
    left_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value], 
                                        landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value], 
                                        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

    # 右手肘角度:右肩[12]、右肘[14]、右腕[16]
    right_elbow_angle = calculate_angle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value], 
                                        landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value], 
                                        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

    # 左手中心點:左腕[15]、左小指[17]、左食指[19]、左姆指[21]
    left_hand_center = np.array([landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_PINKY.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_INDEX.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_THUMB.value]]).mean(axis=0)
    cv2.circle(output_image, (int(left_hand_center[0]), int(left_hand_center[1])), 8, (0, 0, 255), -1)

    # 計算左手中心點與嘴中心點距離
    l_2_m_distance = np.linalg.norm(left_hand_center - mouth_center)

    # 右手中心點:右腕[16]、右小指[18]、右食指[20]、右姆指[22]
    right_hand_center =  np.array([landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_PINKY.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_INDEX.value],
                                    landmarks[mp_pose.PoseLandmark.RIGHT_THUMB.value]]).mean(axis=0)
    cv2.circle(output_image, (int(right_hand_center[0]), int(right_hand_center[1])), 8, (0, 0, 255), -1)

    # 計算右手中心點與嘴中心點距離
    r_2_m_distance = np.linalg.norm(right_hand_center - mouth_center)

    output_image = cv2.flip(output_image, 1)
    
    cv2.putText(output_image, f'Left Elbow Angle: {int(left_elbow_angle)}', (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(output_image, f'L_H to M Distance: {int(l_2_m_distance)}', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(output_image, f'Right Elbow Angle: {int(right_elbow_angle)}', (480, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)    
    cv2.putText(output_image, f'R_H to M Distance: {int(r_2_m_distance)}', (480, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)    

    return output_image, left_elbow_angle, right_elbow_angle, l_2_m_distance, r_2_m_distance