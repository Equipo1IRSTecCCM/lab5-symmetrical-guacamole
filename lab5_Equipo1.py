'''
TE3002B Implementación de robótica inteligente
Equipo 1
    Diego Reyna Reyes A01657387
    Samantha Barrón Martínez A01652135
    Jorge Antonio Hoyo García A01658142
Laboratory 05
Ciudad de México, /05/2022
'''
import cv2
from matplotlib.pyplot import contour
import numpy as np

#Cargar imagenes
img1 = cv2.imread("figs1.png")
img2 = cv2.imread("figs2.png")


#imagen y template a escala de grises
img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#Threshold para imagen y template solo para tener los valores blanco y negro
ret1, thresh1 = cv2.threshold(img1, 176, 255, cv2.THRESH_BINARY_INV)
ret2, thresh2 = cv2.threshold(img2, 176, 255, cv2.THRESH_BINARY_INV)

#encontrar contornos:
edged1 = cv2.Canny(thresh1, 50, 200)
edged2 = cv2.Canny(thresh2, 50, 200)

contours1, hierarchy1 = cv2.findContours(edged1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(edged2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
black = np.zeros(img2.shape)

#ordenar los contornos con su area del mayor a menor:
sortedContours1 = sorted(contours1, key=cv2.contourArea, reverse=True)
sortedContours2 = sorted(contours2, key=cv2.contourArea, reverse=True)

#caolcular los momentos centrales de los contornos:
M1 = cv2.moments(sortedContours1[0])
M2 = cv2.moments(sortedContours2[0])

for c in contours2:
    cv2.drawContours(black, [c], 0, (255,0,0), 3)
    cv2.imshow('Contours by area', black)
    cv2.waitKey(0)
