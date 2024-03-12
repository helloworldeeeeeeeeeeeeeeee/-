import os
import cv2
import numpy as np
import mediapipe as mp

# 初始化MediaPipe人体姿势模型
mp_pose = mp.solutions.pose

# 初始化视频处理对象
mp_drawing = mp.solutions.drawing_utils

def sample_along_n_axis(matrix, num_samples=30):

    n, _,_ = matrix.shape
    if n == num_samples:
        return matrix
    indices = np.linspace(0, n-1, num_samples, dtype=int)
    sampled_matrix = matrix[indices, :]
    return sampled_matrix

# 输入文件夹和输出文件夹路径
input_folder_all = '../test/'
output_folder_all = '../test_npy_02'

for folder in os.listdir(input_folder_all):
    input_folder = os.path.join(input_folder_all,folder)
    output_folder = os.path.join(output_folder_all,folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 初始化Holistic模型
    with mp_pose.Pose(static_image_mode=False,min_detection_confidence=0.1, min_tracking_confidence=0.1) as pose:
        # 循环处理每个视频
        for video_file in os.listdir(input_folder):
            # try:
                if video_file.endswith('.mp4'):
                    video_path = os.path.join(input_folder, video_file)
                    cap = cv2.VideoCapture(video_path)

                    # 初始化存储关键点的列表
                    keypoints_list = []

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break

                        # 进行关键点检测
                        results = pose.process(frame)

                        # 将检测到的关键点数据保存到列表中
                        keypoints_list_item=[]
                        if results.pose_landmarks:
                            pose_landmarks = [(landmark.x, landmark.y) for landmark in results.pose_landmarks.landmark]
                            keypoints_list.append(np.array(pose_landmarks))
                            # left_hand_landmarks = [(landmark.x, landmark.y) for landmark in
                            #                        results.left_hand_landmarks.landmark]
                            # keypoints_list_item.append(np.array(left_hand_landmarks))
                            # right_hand_landmarks = [(landmark.x, landmark.y) for landmark in
                            #                         results.right_hand_landmarks.landmark]
                            # keypoints_list_item.append(np.array(right_hand_landmarks))
                            #
                            # keypoints_list_item = np.concatenate(keypoints_list_item, axis=0)

                        else:
                            continue
                        # keypoints_list.append(np.array(keypoints_list_item))

                        # print(len(pose_landmarks))
                        # print(len(left_hand_landmarks))
                        # print(len(right_hand_landmarks))
                        #
                        # 在图像上绘制关键点
                        annotated_frame = frame.copy()
                        mp_drawing.draw_landmarks(annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        # mp_drawing.draw_landmarks(annotated_frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                        # mp_drawing.draw_landmarks(annotated_frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                        # 显示绘制了关键点的图像
                        # cv2.imshow('Keypoints Detection', annotated_frame)
                        #
                        # if cv2.waitKey(1) & 0xFF == ord('q'):
                        #     break

                    # 释放视频捕获对象
                    cap.release()
                    cv2.destroyAllWindows()

                    # print(len(keypoints_list),len(keypoints_list[0][0]),len(keypoints_list[0][1]),len(keypoints_list[0][2]))
                    keypoints_list = np.array(keypoints_list)
                    print('keypoints_list',keypoints_list.shape)
                    if keypoints_list.shape[0] < 45:
                        continue
                    keypoints_list = sample_along_n_axis(keypoints_list,num_samples=45)
                    print('save keypoints_list',keypoints_list.shape)
                    # 将关键点数据保存为.npy文件
                    np.save(os.path.join(output_folder, f'{video_file.split(".")[0]}_keypoints.npy'), keypoints_list)
            # except:
            #     print(video_file)
            #     continue
