import cv2 as cv
import numpy as np

if __name__ == "__main__":

    slika_bel_kvadrat = np.zeros((512,512), np.uint8)
    slika_bel_kvadrat[100:400,200:300] = 255
    cv.imshow("Beli kvadrat v crni sliki", slika_bel_kvadrat)
    cv.waitKey(0)

    sx = cv.Sobel(slika_bel_kvadrat,cv.CV_64F,1,0,3)
    cv.imshow("Robovi_1", sx)
    cv.imshow("Robovi_2", sx**2)

    cv.waitKey(0)
    cv.destroyAllWindows()