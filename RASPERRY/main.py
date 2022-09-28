from pickle import TRUE
import cv2 as cv
import numpy as np

def resizeImage(src):
     resSrc = cv.resize(src, (200,200), interpolation = cv.INTER_AREA)

     return resSrc

# Encuentra automáticamente el umbral de acuerdo con el método seleccionado
def threshold_demo(image):
         # Imagen en escala de grises
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
         # Imagen binaria
    #ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY | cv.THRESH_TRIANGLE)
    ret, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY)

    print('threshold value %s' % ret)
    cv.imshow('binary', binary)
 
 # Umbral local
def local_threshold(image):

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

     # cv2.adaptiveThreshold(source, max_val, adaptive_method, threshold_type, blocksize, constant)
     # source- It is the source image, which should be a grayscale image.
     # max_val- It specifies the maximum value which is assigned to pixel values exceeding the threshold.
     # adaptive_method- It determines how the threshold value is calculated.
     # threshold_type- It is type of thresholding technique.
     # blocksize- It is the size of a pixel neighbourhood that is used to calculate a threshold value. -> odd number
     # constant- A constant value that is subtracted from the mean or weighted sum of the neighbourhood pixels.

     # blockSize debe ser un número impar, lo siguiente se establece en 25, mayor que el valor promedio 10 (establecido por usted mismo) se establece en blanco o negro, y dentro de 10 se establece en otro color
    dst = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 17)
    dst = resizeImage(dst)

    cv.imshow('binary', dst)    

 # Establecer manualmente el umbral
def threshold_demo_1(image):
     #127
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, binary = cv.threshold(gray, 240, 255, cv.THRESH_TRUNC)
    print('threshold value %s' % ret)
    binary = resizeImage(binary)
    cv.waitKey(0)
    cv.imshow('binary', binary)

############################################

def local_thresholdVideo(videoFrame):
     threshedFrame = cv.adaptiveThreshold(videoFrame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 5, 10)
     #threshedFrame = cv.threshold(videoFrame, 0, 255, cv.THRESH_BINARY)
     return threshedFrame

def videoBinarization():
     capture = cv.VideoCapture(0)

     while True:
          ret, image = capture.read()
          gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
          circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 10)
          threshedFrame = local_thresholdVideo(gray)
          cv.imshow('grayvideo', threshedFrame)

          if cv.waitKey(1) & 0XFF == ord('q'):
               break




def imageBinarization():
     src = cv.imread('./sampleLines/topLeftStraight.jpeg')
     #cv.namedWindow('input image')
     resSrc = resizeImage(src)

     #cv.namedWindow("input image",cv.WINDOW_AUTOSIZE)
     #cv.resizeWindow("input image",(300,300))
     
     cv.imshow('input image', resSrc)

     #threshold_demo(src)
     #custom_threshold(src)
     local_threshold(src)
     cv.waitKey(0)

     cv.destroyAllWindows()

 
     #threshold_demo_1(src)

     if cv.waitKey(1) & 0XFF == ord('q'):
          return

def chooseThresh():
     while(TRUE):
          option = input('1: Image \n2: Video\n')

          if(option == "1"):
               imageBinarization()
          elif(option == "2"):
               videoBinarization()
          else:
               print('Option must be 1 or 2, try again')
               continue



chooseThresh()