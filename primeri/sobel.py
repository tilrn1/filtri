import cv2 as cv
import numpy as np

def sobel_filter(slika, dtype=np.uint8):
    '''Apply Sobel filter to the image.'''
    if dtype is not np.uint8:
        siva = np.float64(cv.cvtColor(slika, cv.COLOR_BGR2GRAY))
        print("HERE")
    else:
        siva = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    zamegljena = cv.GaussianBlur(siva, (3, 3), 0)
    
    # Sobelova jedra
    sobel_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    

    
    
    # Konvolucija slike in obeh filtrov
    gradient_x = cv.filter2D(zamegljena, -1, sobel_x)
    gradient_y = cv.filter2D(zamegljena, -1, sobel_y)

    print("Gradient x\n",gradient_x[0:3,509:512])
    print("Gradient y\n",gradient_y[0:3,509:512])
    
    #Združimo gradienta
    sobel = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    sobel = np.uint8(sobel)
    print(sobel)
    return sobel
    

if __name__ == "__main__":
    slika = cv.imread("../.utils/lenna.png")
    print(slika.shape)
    sobel_1 = sobel_filter(slika,dtype=np.uint8)
    sobel_2 = sobel_filter(slika,dtype=np.float64)
    cv.imshow("Sobel uint8", sobel_1)
    cv.imshow("Sobel float64", sobel_2)
    cv.waitKey(0)
    cv.destroyAllWindows()