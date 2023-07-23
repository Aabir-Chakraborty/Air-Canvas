import cv2
import numpy as np
import pytesseract
from googletrans import Translator
from collections import deque

# Initialize Tesseract OCR and Google Translate
pytesseract.pytesseract.tesseract_cmd = '<path_to_tesseract_executable>'
translator = Translator()

# Define the color ranges for detection (blue, green, red, yellow)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]

# Create trackbars for color selection
cv2.namedWindow("Color detectors")
for i in range(4):
    cv2.createTrackbar(chr(65 + i), "Color detectors", 0, 1, lambda x: None)

# Create a window to display the painting and the frame
paintWindow = np.zeros((471, 636, 3)) + 255
cv2.putText(paintWindow, "CLEAR ALL", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "BLUE", (184, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "GREEN", (276, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "RED", (368, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
cv2.putText(paintWindow, "YELLOW", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)

cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# Load the video
cap = cv2.VideoCapture(0)


# Keep looping
while True:
    # Read the frame
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get the new values of the trackbar in real-time as the user changes them
    u_hue = cv2.getTrackbarPos("Upper Hue", "Color detectors")
    u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color detectors")
    u_value = cv2.getTrackbarPos("Upper Value", "Color detectors")
    l_hue = cv2.getTrackbarPos("Lower Hue", "Color detectors")
    l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color detectors")
    l_value = cv2.getTrackbarPos("Lower Value", "Color detectors")
    Upper_hsv = np.array([u_hue, u_saturation, u_value])
    Lower_hsv = np.array([l_hue, l_saturation, l_value])

    # Detect colors (markers) in the image
    blue = cv2.inRange(hsv, Lower_hsv, Upper_hsv)
    blue = cv2.erode(blue, kernel, iterations=2)
    blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel)
    blue = cv2.dilate(blue, kernel, iterations=1)

    # Find contours of the blue marker
    cnts, _ = cv2.findContours(blue.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    center = None

    if len(cnts) > 0:
        # Find the largest contour and compute its center
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # Process the text inside the marker using Tesseract OCR
        x, y, w, h = cv2.boundingRect(c)
        roi = frame[y:y + h, x:x + w]
        text = pytesseract.image_to_string(roi)

        # Translate the text to English using Google Translate
        translation = translator.translate(text, dest='en')
        translated_text = translation.text

        # Display the translated text on the frame
        cv2.putText(frame, translated_text, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame and the paintWindow image
    cv2.imshow("Tracking", frame)
    cv2.imshow("Paint", paintWindow)

    # Stop if the user presses the Esc key
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
