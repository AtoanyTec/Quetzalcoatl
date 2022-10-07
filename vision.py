import cv2
import numpy

#from djitellopy import tello
#me = tello.Tello()
#me.connect()
#print(me.get_battery())
#me.streamon()
#me.takeoff()

capture = cv2.VideoCapture(0)
HSVValues = [103,10,153,179,255,255]
sensors = 3
threshold = 0.2
width, height = 480, 360
sensitivity = 3
rotationDegrees = [-25, -15, 0, 15, 25]
rotation = 0
forwardSpeed = 15

def thresholding(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lowerWhite = numpy.array([HSVValues[0], HSVValues[1], HSVValues[2]])
    upperWhite = numpy.array([HSVValues[3], HSVValues[4], HSVValues[5]])
    mask = cv2.inRange(hsv, lowerWhite, upperWhite)
    return mask

def getContours(imageThreshold, image):
    xCenter = 0
    contours, hierarchy = cv2.findContours(imageThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if(len(contours) > 0):
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        xCenter = x + w//2
        yCenter = y + h//2
        cv2.drawContours(image, biggest, -1, (255,0,255), 7)
        cv2.circle(image, (xCenter, yCenter), 10, (0, 255, 0), cv2.FILLED)
    return xCenter

def getSensorOutput(imageThreshold, sensors):
    images = numpy.hsplit(imageThreshold, sensors)
    totalPixels = img.shape[1]//sensors * img.shape[0]
    sensorsOutput = []
    for index, image in enumerate(images):
        pixelCount = cv2.countNonZero(image)
        if pixelCount > threshold*totalPixels:
            sensorsOutput.append(1)
        else:
            sensorsOutput.append(0)
        cv2.imshow(str(index), image)
    print(sensorsOutput)
    return sensorsOutput

def sendCommands(sensorsOutput, xCenter):
    global rotation
    translation = (xCenter - width//2)//sensitivity
    translation = int(numpy.clip(translation, -10, 10))
    if translation < 2 and translation > -2: translation = 0
    if sensorsOutput == [0, 0, 0]: rotation = rotationDegrees[2] #rotationDegrees = [-25, -15, 0, 15, 25]
    elif sensorsOutput == [0, 0, 1]: rotation = rotationDegrees[4]
    elif sensorsOutput == [0, 1, 0]: rotation = rotationDegrees[2]
    elif sensorsOutput == [0, 1, 1]: rotation = rotationDegrees[3]
    elif sensorsOutput == [1, 0, 0]: rotation = rotationDegrees[0]
    elif sensorsOutput == [1, 0, 1]: rotation = rotationDegrees[2]
    elif sensorsOutput == [1, 1, 0]: rotation = rotationDegrees[1]
    else: rotation = rotationDegrees[2]
    print(f"{rotation}°")
    #me.send_rc_control(translation, forwardSpeed, 0, rotation)


while True:
    _, img = capture.read()
    #img = me.get_frame_read().frame
    img = cv2.resize(img, (width, height))
    #img = cv2.flip(img, 0)
    imageThreshold = thresholding(img)
    xCenter = getContours(imageThreshold, img) #Traslación
    sensorsOutput = getSensorOutput(imageThreshold, sensors) #Rotation
    sendCommands(sensorsOutput, xCenter)
    cv2.imshow("Output", img)
    cv2.imshow("Path", imageThreshold)
    cv2.waitKey(1)