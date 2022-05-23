import cv2
import numpy as np
import face_recognition

# Langkah Basic

# Mengambil Gambar
imgSyk = face_recognition.load_image_file('images/sykkuno.jpg')
imgSyk = cv2.cvtColor(imgSyk, cv2.COLOR_BGR2RGB)
imgElon = face_recognition.load_image_file('images/elon.jpg')
imgElon = cv2.cvtColor(imgElon, cv2.COLOR_BGR2RGB)

# Menentukan Lokasi dan Mengubah objeck menjadi array
faceLocSyk = face_recognition.face_locations(imgSyk)[0]
encodeSyk = face_recognition.face_encodings(imgSyk)[0]
cv2.rectangle(imgSyk, (faceLocSyk[3], faceLocSyk[0]),
              (faceLocSyk[1], faceLocSyk[2]), (255, 0, 255), 2)

faceLocElon = face_recognition.face_locations(imgElon)[0]
encodeElon = face_recognition.face_encodings(imgElon)[0]
cv2.rectangle(imgElon, (faceLocElon[3], faceLocElon[0]),
              (faceLocElon[1], faceLocElon[2]), (0, 255, 0), 2)

# Menyamakan objek dengan objek lainnya (Compare)
result = face_recognition.compare_faces([encodeElon], encodeSyk)
faceDist = face_recognition.face_distance([encodeElon], encodeSyk)
print(result, faceDist)
cv2.putText(imgElon, f'{result} {round(faceDist[0],2)}', (
    50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

# Menampilkan hasil objeck yang telah di compare
cv2.imshow('Sykkuno', imgSyk)
cv2.imshow('Elon Musk', imgElon)
cv2.waitKey(0)
