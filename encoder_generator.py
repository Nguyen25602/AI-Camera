import cv2
import os
import face_recognition
import pickle
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("ServiceAccount.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': "https://facerealtime-44618-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "facerealtime-44618.appspot.com"
})

# Thêm ảnh người
imgBackground = cv2.imread('Resources/background.png')

# Thêm ID vào StudentIds
folderPath = 'Images'
PathList = os.listdir(folderPath)
imgList = []
studentIds = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    studentIds.append(os.path.splitext(path)[0])

    #Thêm vào Storage firebase
    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

print(studentIds)

def findEncodings(imageList):
    encodeList = []
    for img in imgList:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList
print("Encoding Starting.....")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print("Encoding Complete")

file = open("EncodeFile.p","wb")
pickle.dump(encodeListKnownWithIds, file)
file.close()