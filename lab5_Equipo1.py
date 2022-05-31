'''
TE3002B Implementación de robótica inteligente
Equipo 1
    Diego Reyna Reyes A01657387
    Samantha Barrón Martínez A01652135
    Jorge Antonio Hoyo García A01658142
Laboratory 05
Ciudad de México, /05/2022
'''
#Importar librerias necesarias
import cv2
import numpy as np

def getDeno(M,i,j):
    return M["m00"]**(1+(i+j)/2)

def getDistance(a,b):
    return  abs(b-a)/abs(a)

def calcDistance(A,B):
    distancias = []
    #print(A)
    '''
    for i in range(len(A)):
        t = []
        for j in range(len(B)):
            d = 0
            for k in range(len(B[j])):
                
                if A[i][k] != 0:
                    #print(B[j][k])
                    
                    d += abs(B[j][k]-A[i][k])/abs(A[i][k])
                else:
                    if B[j][k]-A[i][k] == 0:
                        d += 0
                    else:
                        d += abs(B[j][k]-A[i][k])/0.1
            t.append(d)
        distancias.append(t)
    '''
    res = []
    for j in range(len(B)):
        d = 0
        r = 0
        #print(B[j])
        for i in range(7):
            ma = np.sign(A[0][i]) *np.log10(abs(A[0][i]))
            mb = np.sign(B[j][i]) *np.log10(abs(B[j][i]))
            d=abs((ma-mb)/ma)
            if r < d:
                r = d
        res.append(r)
    return res
    #https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/matchcontours.cpp
    #https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/moments.cpp

def getHuMoments(M):
    moments = []
    
    moments.append(M["nu20"]+M["nu02"])
    moments.append((M["nu20"]-M["nu02"])**2 + 4*(M["nu11"])**2)
    moments.append((M["nu30"]-(3*M["nu12"]))**2 + (3*M["nu21"]-M["nu03"])**2)
    moments.append((M["nu30"]+M["nu12"])**2 + (M["nu21"]+M["nu03"])**2)
    moments.append((M["nu30"]-3*M["nu12"])*(M["nu30"]+M["nu12"])*((M["nu30"]+M["nu12"])**2 - 3*(M["nu03"]+M["nu21"])**2) + (3*M["nu21"]-M["nu03"])*(3*(M["nu30"]+M["nu12"])**2 - (M["nu03"]+M["nu21"])**2))
    moments.append((M["nu20"]-M["nu02"])*((M["nu30"]+M["nu12"])**2-(M["nu21"]+M["nu03"])**2)+4*M["nu11"]*(M["nu30"]+M["nu12"])*(M["nu21"]+M["nu03"]))
    moments.append((3*M["nu21"]-M["nu03"])*(M["nu30"]+M["nu12"])*((M["nu30"]+M["nu12"])**2-3*(M["nu21"]+M["nu03"])**2) 
    - (M["nu30"]-3*M["nu12"])*(M["nu21"]+M["nu03"])*(3*(M["nu30"]+M["nu12"])**2 - (M["nu21"]+M["nu03"])**2))
    newMoments = []
    for i in range(len(moments)):
        if abs(moments[i]) != 0:
            newMoments.append(-np.sign(moments[i])*np.log(abs(moments[i])))
        else:
            newMoments.append(0.0)
    return moments
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
edged1 = cv2.Canny(img1, 20, 200)
edged2 = cv2.Canny(img2, 20, 255)
cv2.imshow('2 - All Contours over blank image', edged2)
cv2.waitKey(0)
contours1, hierarchy1 = cv2.findContours(edged1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(img2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2_t = []
#print(contours2)
for c in contours2:
    x,y,w,h = cv2.boundingRect(c)
    tempImg = img2[y:y+h,x:x+w]
    c_temp, hierarchy2 = cv2.findContours(tempImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours2_t.append(c_temp[0])
    #print(cv2.matchShapes(img1, tempImg, cv2.CONTOURS_MATCH_I3,0))
black = np.zeros((img2.shape[0],img2.shape[1],3))
#print(len(contours1))
#ordenar los contornos con su area del mayor a menor:
sortedContours1 = sorted(contours1, key=cv2.contourArea, reverse=True)
sortedContours2 = sorted(contours2_t, key=cv2.contourArea, reverse=True)

#calcular los momentos centrales de los contornos:
M1 = []
M2 = []
hm2 = []
hm1 = []
for i in sortedContours1:
    M1.append(cv2.moments(i))
    hm1.append(getHuMoments(M1[-1]))
for i in sortedContours2:
    M2.append(cv2.moments(i))
    hm2.append(getHuMoments(M2[-1]))
    #print(cv2.HuMoments(cv2.moments(i)))
    #print(hm2[-1])

print()

dist = calcDistance(hm1,hm2)
print(dist)
val = 7.0

for j in range(len(dist)):
    if dist[j] <= val:
        #print(dist[i][j])
        c = contours2[j]
        cv2.drawContours(black, [c], 0, (0,255,0), 3)
        cv2.imshow('Contours by area', black)
cv2.waitKey(0)
'''
for c in contours1:
    cv2.drawContours(black, [c], 0, (0,255,0), 3)
    cv2.imshow('Contours by area', black)
    cv2.waitKey(0)
'''

