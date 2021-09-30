#import the required libraries
import cv2
import matplotlib.pyplot as plt

#Read and print the given image
image=cv2.imread(r"C:\Users\Public\Pictures\Sample Pictures\Desert.jpg")
plt.imshow(image[:,:,::-1])
plt.axis(False)
plt.title("Original Image")
plt.show()

#Convert the given image into grayscale image
grayscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
plt.imshow(grayscale,cmap='gray')
plt.axis(False)
plt.title("Grayscale Image")
plt.show()

#Invert the grayscale image 
invert=cv2.bitwise_not(grayscale)
plt.imshow(invert,cmap='gray')
plt.axis(False)
plt.title("Inverted Image")
plt.show()

#Smoothen the Inverted image
imgsmooth=cv2.GaussianBlur(invert,(21,21),sigmaX=0,sigmaY=0)
plt.imshow(imgsmooth,cmap='gray')
plt.axis(False)
plt.title("Smoothened Image")
plt.show()

#Print the final result(A Pencil Sketch of the given image)
final = cv2.divide(grayscale,255-imgsmooth,scale=255)
plt.imshow(final,cmap='gray')
plt.axis(False)
plt.title("Final Image")
plt.show()
