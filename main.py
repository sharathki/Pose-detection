import cv2
import mediapipe as mp
import numpy as np

mpose=mp.solutions.pose
mpDraw=mp.solutions.drawing_utils
pose=mpose.Pose()



cap=cv2.VideoCapture('4.mp4')

drawspec1=mpDraw.DrawingSpec(thickness=3,circle_radius=3,color=(0,0,255))
drawspec2=mpDraw.DrawingSpec(thickness=3,circle_radius=3,color=(0,255,0))

while True:
    success,img=cap.read()
    img=cv2.resize(img,(1200,1000))
    results=pose.process(img)
    mpDraw.draw_landmarks(img,results.pose_landmarks,mpose.POSE_CONNECTIONS,drawspec1,drawspec2)
#creating blank pose with marks
    h,w,c =img.shape
    img2=np.zeros([h,w,c])
    img2.fill(255)
    mpDraw.draw_landmarks(img2, results.pose_landmarks, mpose.POSE_CONNECTIONS, drawspec1, drawspec2)

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cv2.imshow('PoseDetection',img)
    cv2.imshow('blank pose', img2)

    cv2.waitKey(1)