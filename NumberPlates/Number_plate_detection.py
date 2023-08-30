import cv2
import os
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

frameWidth = 640
franeHeight = 480

plateCascade = cv2.CascadeClassifier("D:\\Projects\\License plate Detection\\haarcascade_russian_plate_number.xml")


minArea = 500

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, franeHeight)
cap.set(10, 150)
count = 0

state_mapping = {
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JK": "Jammu and Kashmir",
    "JH": "Jharkhand",
    "KA": "Karnataka",
    "KL": "Kerala",
    "MP": "Madhya Pradesh",
    "MH": "Maharashtra",
    "MN": "Manipur",
    "ML": "Meghalaya",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TS": "Telangana",
    "TR": "Tripura",
    "UP": "Uttar Pradesh",
    "UK": "Uttarakhand",
    "WB": "West Bengal",
    "AN": "Andaman and Nicobar Islands",
    "CH": "Chandigarh",
    "DN": "Dadra and Nagar Haveli and Daman and Diu",
    "DL": "Delhi",
    "LD": "Lakshadweep",
    "PY": "Puducherry"
}

while True:
    success, img = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade .detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img,"Number Plate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            imgRoi = img[y:y+h, x:x+w]
            imgRoi_gray = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
            plate_text = pytesseract.image_to_string(imgRoi_gray, config='--psm 6').strip()

            cv2.imshow("ROI",imgRoi)
            if plate_text[0:2] in state_mapping:
                detected_state = state_mapping[plate_text[0:2]]
                cv2.putText(img, f"State: {detected_state}", (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
                print("State:", detected_state)
    cv2.imshow("Result", img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        filename = os.path.join(r"D:\Projects\License plate Detection\number-plates", f"plate-{count}.jpg")

        while os.path.exists(filename):
            count += 1
            filename = os.path.join(r"D:\Projects\License plate Detection\number-plates", f"plate-{count}.jpg")

        cv2.imwrite(filename, imgRoi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1

    elif key == ord('b'):
        break
cap.release()
cv2.destroyAllWindows()
