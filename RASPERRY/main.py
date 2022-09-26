import cv2 as cv
import numpy as np



 
# Encuentra automáticamente el umbral de acuerdo con el método seleccionado
def threshold_demo(image):
         # Imagen en escala de grises
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
         # Imagen binaria
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    print('threshold value %s' % ret)
    cv.imshow('binary', binary)
 
 
 # Umbral local
def local_threshold(image):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    

     # blockSize debe ser un número impar, lo siguiente se establece en 25, mayor que el valor promedio 10 (establecido por usted mismo) se establece en blanco o negro, y dentro de 10 se establece en otro color
    dst = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 25, 10)
    cv.imshow('binary', dst)
 
 
 # Umbral adaptativo
def custom_threshold(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #h, w = gray.shape[:2]
    #print(w)
    #m = np.reshape(gray, [1, w*h])
         # Mean
    #mean = m.sum() / (w*h)
    #print('mean:', mean)
    ret, binary = cv.threshold(gray, 220, 255, cv.THRESH_BINARY)
    cv.namedWindow("binary",cv.WINDOW_NORMAL)

    cv.resizeWindow("binary",300,300)
    cv.imshow('binary', binary)
 
 
 # Establecer manualmente el umbral
def threshold_demo_1(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)
    print('threshold value %s' % ret)
    cv.imshow('binary', binary)
 
 
src = cv.imread('./sampleLines/topLeftStraight.jpeg')
cv.namedWindow('input image')

cv.namedWindow("input image",cv.WINDOW_NORMAL)

cv.resizeWindow("input image",300,300)
cv.imshow('input image', src)

#
custom_threshold(src)
#local_threshold(src)
#threshold_demo_1(src)

#
cv.waitKey(0)
cv.destroyAllWindows()


