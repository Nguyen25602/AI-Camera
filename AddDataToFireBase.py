import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("ServiceAccount.json")
firebase_admin.initialize_app(cred,{
    'databaseURL': "https://facerealtime-44618-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

ref = db.reference('Students')

data = {
    "10052005":
        {
            "name": "Phuong Anh",
            "major": "My Love",
            "starting_year": 2023,
            "total_attendance": 1,
            "standing": "SSS",
            "year": 100,
            "last_attendance_time": "2023-06-13 17:06:00"
        }
}

for key,value in data.items():
    ref.child(key).set(value)