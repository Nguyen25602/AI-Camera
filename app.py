import os
import cv2
import pickle
import numpy as np
import face_recognition
import cvzone
import firebase_admin
import argparse
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db
from utils import CvFpsCalc
from datetime import datetime

cred = credentials.Certificate("ServiceAccount.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': "https://facerealtime-44618-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "facerealtime-44618.appspot.com"
})

bucket = storage.bucket()

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=640)
    parser.add_argument("--height", help='cap height', type=int, default=480)

    args = parser.parse_args()

    return args

def main():
    # Argument parsing #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    # Camera preparation ###############################################################
    cap = cv2.VideoCapture(cap_device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cap_height)

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=1)

    # Thêm Background
    imgBackground = cv2.imread('Resources/background.png')

    # Thêm Modes
    folderModePath = 'Resources/Modes'
    modePathList = os.listdir(folderModePath)
    imgModeList = []
    for path in modePathList:
        imgModeList.append(cv2.imread(os.path.join(folderModePath,path)))

    # Thêm encoding file
    print("Loading Encode File ...")
    file = open('EncodeFile.p','rb')
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    encodeListKnown, studentIds = encodeListKnownWithIds
    print("Encode File Complete")
    # print(studentIds)

    # Trạng thái của System
    modeType = 0
    counter = 0
    id = -1
    imgStudent = []

    while True:
        fps = cvFpsCalc.get()
        # Process Key (ESC: end) #################################################
        key = cv2.waitKey(10)
        if key == 27:  # ESC
            break
        success, img = cap.read()
        if not success:
            break

        imgS = cv2.resize(img, (0,0), None,0.25,0.25)
        imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS, model="hog")
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        imgBackground[200:200 + 480, 90:90 + 640] = img
        imgBackground[50:50 + 688, 860:860 + 513] = imgModeList[modeType]

        if faceCurFrame:
            for encodeFace, faceLoc in zip(encodeCurFrame,faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
                # print("Matches",matches)
                # print("faceDis",faceDis)

                matchIndex = np.argmin(faceDis)
                # print("Match Index",matchIndex)
                if matches[matchIndex]:
                    # print("Phát Hiện Khuôn Mặt")
                    # print(studentIds[matchIndex])
                    y1, x2, y2, x1 = faceLoc
                    bbox = 90 + x1, 200 + y1, x2 - x1, y2 - y1
                    imgBackground= cvzone.cornerRect(imgBackground, bbox, rt=0)
                    id = studentIds[matchIndex]
                    # Mode Tìm Kiếm
                    if counter == 0:
                        counter = 1
                        modeType = 1 # Chuyển trạng thái Searching
            if counter != 0:
                if counter == 1:
                    studentInfo = db.reference(f'Students/{id}').get()
                    print(studentInfo)
                    # Lấy image từ Cloud
                    blob = bucket.get_blob(f'Images/{id}.png')
                    array = np.frombuffer(blob.download_as_string(),np.uint8)
                    imgStudent = cv2.imdecode(array,cv2.COLOR_BGRA2BGR)
                    # Update data sau 3p
                    datatimeObject = datetime.strptime(studentInfo['last_attendance_time'],
                                                      "%Y-%m-%d %H:%M:%S")
                    secondsElapsed = (datetime.now()-datatimeObject).total_seconds()
                    # print(secondsElapsed)
                    if secondsElapsed >30:
                        ref = db.reference(f'Students/{id}')
                        studentInfo['total_attendance'] += 1
                        ref.child('total_attendance').set(studentInfo['total_attendance'])
                        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        modeType = 2
                        counter = 0
                        imgBackground[50:50 + 688, 860:860 + 513] = imgModeList[modeType]
                # Đếm số lần để chuyển ModeType
                if modeType != 2:
                    if 10<counter<20:
                        modeType = 3

                    imgBackground[50:50 + 688, 860:860 + 513] = imgModeList[modeType]

                    if counter <= 10:
                        # Thêm Ngành
                        cv2.putText(imgBackground, str(studentInfo['major']), (1045, 135),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                        # Thêm số lần
                        cv2.putText(imgBackground,str(studentInfo['total_attendance']),(930,135),
                                    cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),1)
                        # Thêm MSSV
                        cv2.putText(imgBackground, str(id), (1100, 535),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)
                        # Thêm Tên
                        cv2.putText(imgBackground, str(studentInfo['name']), (1100, 600),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 0), 1)
                        # Thêm Điểm Đánh Giá
                        cv2.putText(imgBackground, str(studentInfo['standing']), (955, 700),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                        # Thêm Điểm Năm Học
                        cv2.putText(imgBackground, str(studentInfo['year']), (1110, 700),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                        # Thêm Năm Bắt Đầu
                        cv2.putText(imgBackground, str(studentInfo['starting_year']), (1265, 700),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                        imgBackground[170:170+300, 967:967+300] = imgStudent
                    counter += 1

                    if counter >= 20:
                        counter = 0
                        modeType = 0
                        studentInfo = []
                        imgStudent = []
                        imgBackground[50:50 + 688, 860:860 + 513] = imgModeList[modeType]
        else:
            modeType=0
            counter=0
        cv2.imshow("Face Attendance", imgBackground)
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()