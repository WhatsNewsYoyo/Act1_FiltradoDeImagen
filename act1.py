import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

images =["placa_q.jpg", "placa_4.jpg"]
n = len(images)
for i in range (0, n) :
    img = cv2.imread(images[i])
    negative_image = 255 - img  
    hsv = cv2.cvtColor(negative_image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 40, 255])

    mask_white = cv2.inRange(hsv, lower_white, upper_white)
    result = cv2.bitwise_and(negative_image, negative_image, mask=mask_white)

    #Gaussian
    gaussian = cv2.GaussianBlur(result, (3,3), 1)

    kernel = np.ones((10,10), np.uint8) 
    dilate = cv2.dilate(gaussian, kernel, iterations=1)
    dilate_gray = cv2.cvtColor(dilate, cv2.COLOR_BGR2GRAY)

    #Reader
    texto = pytesseract.image_to_string(dilate_gray)
    print(texto)

    plt.figure(figsize=(1,10))
    plt.subplot(1,10,1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,10,2)
    plt.imshow(cv2.cvtColor(negative_image, cv2.COLOR_BGR2RGB))
    plt.title("Negative")
    plt.axis('off')

    plt.subplot(1,10,3)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("White filter")
    plt.axis('off')

    plt.subplot(1,10,4)
    plt.imshow(cv2.cvtColor(gaussian, cv2.COLOR_BGR2RGB))
    plt.title("Gaussian")
    plt.axis('off')

    plt.subplot(1,10,5)
    plt.imshow(cv2.cvtColor(dilate, cv2.COLOR_BGR2RGB))
    plt.title("Dilate")
    plt.axis('off')

    plt.subplot(1,10,6)
    plt.imshow(cv2.cvtColor(dilate_gray, cv2.COLOR_BGR2RGB))
    plt.title("Gray")
    plt.axis('off')
    
    plt.show()
