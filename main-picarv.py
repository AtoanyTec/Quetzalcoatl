import cv2
import numpy

def thresholding(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lowerWhite = numpy.array([HSVValues[0], HSVValues[1], HSVValues[2]])
    upperWhite = numpy.array([HSVValues[3], HSVValues[4], HSVValues[5]])
    mask = cv2.inRange(hsv, lowerWhite, upperWhite)
    return mask

def getContours(imageThreshold, image):
    xCenter, x, w = 0, 0, 0
    contours, hierarchy = cv2.findContours(imageThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if(len(contours) > 0):
        biggest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(biggest)
        xCenter = x + w//2
        yCenter = y + h//2
        cv2.drawContours(image, biggest, -1, (255,0,255), 7)
        cv2.circle(image, (xCenter, yCenter), 10, (0, 255, 0), cv2.FILLED)
    return [xCenter, x, w]

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
        image = cv2.resize(image, (width//sensors, height))
        cv2.imshow(str(index), image)
    print(sensorsOutput)
    return sensorsOutput

def sendCommands(sensorsOutput, xCenter):
    global rotation
    translation = (xCenter - width//2)//sensitivity
    translation = int(numpy.clip(translation, -10, 10))
    if translation < 2 and translation > -2:
        translation = 0

    if sensorsOutput == [0, 0, 0, 0, 0]:
        rotation = rotationDegrees[4]
    elif sensorsOutput == [0, 0, 0, 0, 1]:
        rotation = rotationDegrees[8]
    elif sensorsOutput == [0, 0, 0, 1, 0]:
        rotation = rotationDegrees[7]
    elif sensorsOutput == [0, 0, 0, 1, 1]:
        rotation = rotationDegrees[8]
    elif sensorsOutput == [0, 0, 1, 0, 0]:
        rotation = rotationDegrees[4]
    elif sensorsOutput == [0, 0, 1, 0, 1]:
        rotation = rotationDegrees[6]
    elif sensorsOutput == [0, 0, 1, 1, 0]:
        rotation = rotationDegrees[5]
    elif sensorsOutput == [0, 0, 1, 1, 1]:
        rotation = rotationDegrees[6]
    elif sensorsOutput == [0, 1, 0, 0, 0]:
        rotation = rotationDegrees[1]
    elif sensorsOutput == [0, 1, 0, 0, 1]:
        rotation = rotationDegrees[5]
    elif sensorsOutput == [0, 1, 0, 1, 0]:
        rotation = rotationDegrees[4]
    elif sensorsOutput == [0, 1, 0, 1, 1]:
        rotation = rotationDegrees[5]
    elif sensorsOutput == [0, 1, 1, 0, 0]:
        rotation = rotationDegrees[3]
    elif sensorsOutput == [0, 1, 1, 0, 1]:
        rotation = rotationDegrees[5]
    elif sensorsOutput == [0, 1, 1, 1, 0]:
        rotation = rotationDegrees[4]
    elif sensorsOutput == [0, 1, 1, 1, 1]:
        rotation = rotationDegrees[5]
    elif sensorsOutput == [1, 0, 0, 0, 0]:
        rotation = rotationDegrees[0]
    elif sensorsOutput == [1, 0, 0, 0, 1]:
        rotation = rotationDegrees[4]
    elif sensorsOutput == [1, 0, 0, 1, 0]:
        rotation = rotationDegrees[3]
    elif sensorsOutput == [1, 0, 0, 1, 1]:
        rotation = rotationDegrees[5]
    elif sensorsOutput == [1, 0, 1, 0, 0]:
        rotation = rotationDegrees[2]
    elif sensorsOutput == [1, 0, 1, 0, 1]:
        rotation = rotationDegrees[4]
    elif sensorsOutput == [1, 0, 1, 1, 0]:
        rotation = rotationDegrees[3]
    elif sensorsOutput == [1, 0, 1, 1, 1]:
        rotation = rotationDegrees[5]
    elif sensorsOutput == [1, 1, 0, 0, 0]:
        rotation = rotationDegrees[0]
    elif sensorsOutput == [1, 1, 0, 0, 1]:
        rotation = rotationDegrees[3]
    elif sensorsOutput == [1, 1, 0, 1, 0]:
        rotation = rotationDegrees[3]
    elif sensorsOutput == [1, 1, 0, 1, 1]:
        rotation = rotationDegrees[4]
    elif sensorsOutput == [1, 1, 1, 0, 0]:
        rotation = rotationDegrees[2]
    elif sensorsOutput == [1, 1, 1, 0, 1]:
        rotation = rotationDegrees[3]
    elif sensorsOutput == [1, 1, 1, 1, 0]:
        rotation = rotationDegrees[3]
    elif sensorsOutput == [1, 1, 1, 1, 1]:
        rotation = rotationDegrees[4]
    print(f"{rotation}°")

def findCircle(imageThreshold):
    circle = None
    imageThreshold = cv2.blur(imageThreshold, (11, 11))
    detected_circles = cv2.HoughCircles(imageThreshold, cv2.HOUGH_GRADIENT, 1, 150, param1=60, param2=50, minRadius=15, maxRadius=285)
    if detected_circles is not None:
        detected_circles = numpy.uint16(numpy.around(detected_circles))
        circle = detected_circles[0, numpy.argmax(numpy.uint16(numpy.around(detected_circles[:, :, 2])))]
        cv2.circle(img, (circle[0], circle[1]), circle[2], (0, 255, 0), 7)
        cv2.circle(img, (circle[0], circle[1]), 1, (0, 0, 255), 3)
    return circle

def empty(a):
    pass

def colorPicker():
    global HSVValues
    cv2.namedWindow("HSV")
    cv2.resizeWindow("HSV", 640, 240)
    cv2.createTrackbar("HUE Min", "HSV", 0, 179, empty)
    cv2.createTrackbar("HUE Max", "HSV", 179, 179, empty)
    cv2.createTrackbar("SAT Min", "HSV", 0, 255, empty)
    cv2.createTrackbar("SAT Max", "HSV", 255, 255, empty)
    cv2.createTrackbar("VALUE Min", "HSV", 0, 255, empty)
    cv2.createTrackbar("VALUE Max", "HSV", 255, 255, empty)
    while True:
        rtf, img = capture.read()
        img = cv2.resize(img, (width, height))
        imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_min = cv2.getTrackbarPos("HUE Min", "HSV")
        h_max = cv2.getTrackbarPos("HUE Max", "HSV")
        s_min = cv2.getTrackbarPos("SAT Min", "HSV")
        s_max = cv2.getTrackbarPos("SAT Max", "HSV")
        v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
        v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
        lower = numpy.array([h_min, s_min, v_min])
        upper = numpy.array([h_max, s_max, v_max])
        mask = cv2.inRange(imgHsv, lower, upper)
        result = cv2.bitwise_and(img, img, mask=mask)
        HSVValues = [h_min, s_min, v_min, h_max, s_max, v_max]
        print(HSVValues)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        hStack = numpy.hstack([img, mask, result])
        cv2.imshow('Horizontal Stacking', hStack)
        if cv2.waitKey(1) & 0xFF == ord('\u001B'):
            break
    cv2.destroyAllWindows()

capture = cv2.VideoCapture(0)
HSVValues = []
sensors = 5
threshold = 0.2
width, height = 480, 360
sensitivity = 3
rotationDegrees = numpy.linspace(start=-25, stop=25, num=sensors*2-1)
rotation = 0
forwardSpeed = 15

colorPicker()
while True:
    print(rotationDegrees)
    rtf, img = capture.read()
    if not rtf:
        break
    img = cv2.resize(img, (width, height))
    imageThreshold = thresholding(img)
    xCenter, x, w = getContours(imageThreshold, img)
    sensorsOutput = getSensorOutput(imageThreshold, sensors)
    sendCommands(sensorsOutput, xCenter)
    circle = findCircle(imageThreshold)
    print(circle)
    if circle is not None and circle[2]*2 > w:
        print("Posicionar")
    else:
        print("No cuenta")
    cv2.imshow("Output", img)
    cv2.imshow("Path", imageThreshold)
    if cv2.waitKey(1) & 0xFF == ord('\u001B'):
        break
capture.release()
cv2.destroyAllWindows()