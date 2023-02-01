#Importar librerias necesarias
import cv2
import numpy as np

#Lectura de imagen y selección de archivos Haar 
cap = cv2.VideoCapture(0)
#fgbg = cv2.createBackgroundSubtractorMOG2()

stop_sign = cv2.CascadeClassifier(r"C:\Users\ignac\OneDrive\Escritorio\Python\TFG\XMLs\cascade_stop_sign.xml")
noturnleft_sign = cv2.CascadeClassifier(r"C:\Users\ignac\OneDrive\Escritorio\Python\TFG\XMLs\no_turn_left.xml")
noturnright_sign = cv2.CascadeClassifier(r"C:\Users\ignac\OneDrive\Escritorio\Python\TFG\XMLs\no_turn_right.xml")
#stop_sign = cv2.CascadeClassifier(r"C:\Users\ignac\OneDrive\Escritorio\Python\TFG\haar_cascade_train\stop-\classifier\cascade_stop.xml")
#noturnleft_sign = cv2.CascadeClassifier(r"C:\Users\ignac\OneDrive\Escritorio\Python\TFG\haar_cascade_train\giro izquierda-\classifier\cascade.xml")
#noturnright_sign = cv2.CascadeClassifier(r"C:\Users\ignac\OneDrive\Escritorio\Python\TFG\haar_cascade_train\giro derecha-\classifier\cascade.xml")
caution_sign = cv2.CascadeClassifier(r"C:\Users\ignac\OneDrive\Escritorio\Python\TFG\XMLs\caution.xml")
right_sign = cv2.CascadeClassifier(r"C:\Users\ignac\OneDrive\Escritorio\Python\TFG\XMLs\right.xml")
left_sign = cv2.CascadeClassifier(r"C:\Users\ignac\OneDrive\Escritorio\Python\TFG\XMLs\left.xml")
yield_sign = cv2.CascadeClassifier(r"C:\Users\ignac\OneDrive\Escritorio\Python\TFG\XMLs\yield.xml")
parking_sign = cv2.CascadeClassifier(r"C:\Users\ignac\OneDrive\Escritorio\Python\TFG\XMLs\parking.xml")

while cap.isOpened(): 


	#Definir variables y leer el primer frame
	ret, frame = cap.read()
	#fgbg.apply(frame)

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	color_green = (0,255,0) 
	color_yellow = (0,255,255)

	#Modelo de Haar Cascade
	stop_sign_scaled = stop_sign.detectMultiScale(gray, 1.2, minNeighbors=10 )
	noturnleft_sign_scaled = noturnleft_sign.detectMultiScale(gray, 1.1, minNeighbors=4 )
	noturnright_sign_scaled = noturnright_sign.detectMultiScale(gray, 1.1, minNeighbors=3 )
	caution_sign_scaled = caution_sign.detectMultiScale(gray, 1.1, minNeighbors=3 )
	right_sign_scaled = right_sign.detectMultiScale(gray, 1.1, minNeighbors=3 )
	left_sign_scaled = left_sign.detectMultiScale(gray, 1.1, 3)
	yield_sign_scaled = yield_sign.detectMultiScale(gray, 1.1, 3)
	parking_sign_scaled = parking_sign.detectMultiScale(gray, 1.1, 3)

	#Dibuja rectangulo en las coordenadas del objeto y texto encima del rectangulo identificando su clase

	for (x,y,w,h) in stop_sign_scaled:
		cv2.rectangle(frame, (x,y), (x+w, y+h), color_green, 2)
		cv2.putText(frame, "stop", (x,y-5), 1, 1.2, color_yellow, 2)

	for (x,y,w,h) in noturnleft_sign_scaled:
		cv2.rectangle(frame, (x,y), (x+w, y+h), color_green, 2)
		cv2.putText(frame, "no turn left", (x,y-5), 1, 1.2, color_yellow, 2)
	
	for (x,y,w,h) in noturnright_sign_scaled:
		cv2.rectangle(frame, (x,y), (x+w, y+h), color_green, 2)
		cv2.putText(frame, "no turn right", (x,y-5), 1, 1.2, color_yellow, 2)

	for (x,y,w,h) in caution_sign_scaled:
		cv2.rectangle(frame, (x,y), (x+w, y+h), color_green, 2)
		cv2.putText(frame, "caution", (x,y-5), 1, 1.2, color_yellow, 2)

	for (x,y,w,h) in right_sign_scaled:
		cv2.rectangle(frame, (x,y), (x+w, y+h), color_green, 2)
		cv2.putText(frame, "right", (x,y-5), 1, 1.2, color_yellow, 2)

	for (x,y,w,h) in left_sign_scaled:
		cv2.rectangle(frame, (x,y), (x+w, y+h), color_green, 2)
		cv2.putText(frame, "left", (x,y-5), 1, 1.2, color_yellow, 2)		

	for (x,y,w,h) in yield_sign_scaled:
		cv2.rectangle(frame, (x,y), (x+w, y+h), color_green, 2)
		cv2.putText(frame, "yield", (x,y-5), 1, 1.2, color_yellow, 2)	

	for (x,y,w,h) in parking_sign_scaled:
		cv2.rectangle(frame, (x,y), (x+w, y+h), color_green, 2)
		cv2.putText(frame, "parking", (x,y-5), 1, 1.2, color_yellow, 2)	

	#Imagen por pantalla
	#cv2.imshow('BackgroundSubtractor',fgbg)
	#cv2.imshow('Gray',gray)
	cv2.imshow("Webcam On", frame)

	#Finalización del proceso
	if cv2.waitKey(30) == 27:
		break

cap.release()
cv2.destroyAllWindows()
	