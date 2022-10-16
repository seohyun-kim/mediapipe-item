import cv2
import time
import mediapipe as mp
from math import hypot
import imutils

# 영상 및 돼지코 이미지 로드
cap = cv2.VideoCapture(0)
hair_img = cv2.imread('./img/hair3.png')

# 5개의 center nose landmark point
hair_landmarks = [127, 356, 10, 9, 10]


# mediapipe 호출
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=4)  # max_num_faces로 영상에서 인식할 얼굴 개수 설정

# 특정 조건이 충족될때까지 영상 재생 esc버튼 종료시까지
while True:
    # 영상 읽기
    ret, frame = cap.read()
    # frame 크기 조정
    frame = imutils.resize(frame, width=500)
    # frame에서 facemesh 검출
    results = faceMesh.process(frame)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # nose landmarks
            left_hair_x = 0
            left_hair_y = 0
            right_hair_x = 0
            right_hair_y = 0
            center_hair_x = 0
            center_hair_y = 0

            # 각 랜드마크의 정보 가져오기
            for id, lm in enumerate(face_landmarks.landmark):
                # frame의 height, width, channel
                h, w, c = frame.shape
                # 랜드마크의 x좌표와 frame의 width를 곱해주어 x로 설정,랜드마크의 y좌표와 frame의 height을 곱하여 y좌표
                x, y = int(lm.x * w), int(lm.y * h)

                # 앞서 설정한 nose_landmark와 일치하는 랜드마크 넘버에 대해 x,y좌표 부여
                if id == hair_landmarks[0]:
                    left_hair_x, left_hair_y = x, y
                if id == hair_landmarks[1]:
                    right_hair_x, right_hair_y = x, y
                if id == hair_landmarks[4]:
                    center_hair_x, center_hair_y = x, y

            # hair_width 계산
            hair_width = int(hypot(left_hair_x - right_hair_x, left_hair_y - right_hair_y * 1.2))
            hair_height = int(hair_width * 0.77)

            # nose_width와 nose_height가 0이 아닐 때 돼지코 이미지를 해당 크기에 맞게 resize
            if (hair_width and hair_height) != 0:
                hair_sample = cv2.resize(hair_img, (hair_width, hair_height))

            # nose_area 구하기
            top_left = (int(center_hair_x - hair_width / 2), int(center_hair_y - hair_height / 2))
            bottom_right = (int(center_hair_x + hair_width / 2), int(center_hair_y + hair_height / 2))

            nose_area = frame[
                        top_left[1]: top_left[1] + hair_height,
                        top_left[0]: top_left[0] + hair_width
                        ]

            # nose mask 생성
            hair_sample_gray = cv2.cvtColor(hair_sample, cv2.COLOR_BGR2GRAY)
            _, nose_mask = cv2.threshold(hair_sample_gray, 25, 255, cv2.THRESH_BINARY_INV)
            no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)

            # no_nose에 pig nose 중첩
            final_nose = cv2.add(no_nose, hair_sample)
            # pig nose filter를 영상에 적용
            frame[
            top_left[1]: top_left[1] + hair_height,
            top_left[0]: top_left[0] + hair_width
            ] = final_nose

    # 변경된 이미지 출력
    cv2.imshow("output", frame)
    # esc 입력시 종료
    key = cv2.waitKey(10)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
