'''
TE3002B Implementación de robótica inteligente
Equipo 1
    Diego Reyna Reyes A01657387
    Samantha Barrón Martínez A01652135
    Jorge Antonio Hoyo García A01658142
Laboratory 05
Ciudad de México, 31/05/2022
'''
#Importar librerias necesarias
import cv2
import numpy as np


def calcDistance(A,B):
    distancias = []
    for j in range(len(B)):
        d = 0
        t = []
        for k in range(len(B[j])):
            
            if A[0][k] != 0:
                #print(B[j][k])
                
                t.append(abs(B[j][k]-A[0][k])/abs(A[0][k]))
            else:
                if B[j][k]-A[0][k] == 0:
                    t.append(0)
                else:
                    t.append(abs(B[j][k]-A[0][k])/0.1)
        
        distancias.append(t)
    return distancias


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
    return newMoments
#Cargar imagenes
img1 = cv2.imread("figs1.png")
img2 = cv2.imread("figs2.png")


#imagen y template a escala de grises
img1_g = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img2_g = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#Threshold para imagen y template solo para tener los valores blanco y negro
ret1, thresh1 = cv2.threshold(img1_g, 254, 255, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(img2_g, 254, 255, cv2.THRESH_BINARY)

#Sacar contornos
contours1, hierarchy1 = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
contours2, hierarchy2 = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

#ordenar los contornos con su area del mayor a menor:
sortedContours1 = sorted(contours1, key=cv2.contourArea, reverse=True)
sortedContours2 = sorted(contours2, key=cv2.contourArea, reverse=True)

#calcular los momentos centrales y los Hu moments de los contornos:
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

#Calcular las distancias de los momentos
dist = calcDistance(hm1,hm2)
val = 0.1
c = 0
for di in dist:
    if di[0] < val:
        print(di[0])
        co = sortedContours2[c]
        #Imprimir el contorno si son parecidas
        cv2.drawContours(img2, [co], 0, (0,255,0), 3)
    #Calcular el centro de masa 
    cm = (int(M2[c]['m10']/M2[c]['m00']),int(M2[c]['m01']/M2[c]['m00']))
    cv2.circle(img2,cm,3,(0,255,0),2)
    #Calcular la elipse equivalente
    cov = np.asarray([[M2[c]['mu20'], M2[c]['mu11']],
                        [M2[c]['mu11'], M2[c]['mu20']]])
    eigvalues, eigvectors = np.linalg.eig(cov)
    eigval_1, eigval_2 = eigvalues
    eigvec_1, eigvec_2 = eigvectors[:, 0], eigvectors[:, 1]
    theta = np.arctan2(eigvec_1[1], eigvec_1[0])
    angle = np.rad2deg(theta)

    a = np.sqrt((eigval_1/M2[c]['m00'])) * 2
    b = np.sqrt((eigval_2/M2[c]['m00'])) * 2
    print(a,b)
    r = a/b
    #Grax: https://notebook.community/hadim/public_notebooks/Analysis/Fit_Ellipse/notebook
    #Imprimit en imagen
    cv2.ellipse(img2, cm, (int(a), int(b)),angle,0,360,(0,0,255),3)
    cv2.putText(img2, str(round(r,4)), cm, cv2.FONT_HERSHEY_TRIPLEX, 0.9, (0,0,255), 1, cv2.LINE_AA)
    c+=1
cv2.imshow('Yo y mis hermanos', img2)
cv2.waitKey(0)


