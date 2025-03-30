import cv2 as cv
import numpy as np

if __name__ == '__main__':
    #slika = cv.imread('../.utils/lenna.png')
    #slika = cv.cvtColor(slika, cv.COLOR_BGR2GRAY)
    
    slike = list()
    slika1 = np.zeros((4, 4), dtype=np.uint8)
    slika2 = np.zeros((4, 4), dtype=np.uint8)
    slika3 = np.zeros((4, 4), dtype=np.uint8)
    
    ############################################
    #Prva slika
    slika1[0, 0] = 1
    slika1[1, 1] = 2
    slika1[2, 2] = 3    
    slika1[3, 3] = 4
    ###########
    # 1 0 0 0 #
    # 0 2 0 0 #
    # 0 0 3 0 #
    # 0 0 0 4 #
    ###########

    ############################################
    #Druga slika
    slika2[0, 0] = 1
    slika2[1, 1] = 1
    slika2[2, 2] = 1    
    slika2[3, 3] = 1
    ###########
    # 1 0 0 0 #
    # 0 1 0 0 #
    # 0 0 1 0 #
    # 0 0 0 1 #
    ###########

    ############################################
    #Tretja slika
    slika3[1, 1] = 1

    ###########
    # 0 0 0 0 #
    # 0 1 0 0 #
    # 0 0 0 0 #
    # 0 0 0 0 #
    ###########
    slike = [slika1, slika2,slika3]

    #Izvedemo konvolucijo slike z jedrom
    # Najprej definiramo jedro
    jedro = np.array([[1, 1, 1],
                     [1, 1, 1],
                     [1, 1, 1]])
    
    # Izvedemo konvolucijo
    for slika in slike:
        slika_konvolucija = cv.filter2D(slika, -1, jedro,borderType=cv.BORDER_CONSTANT)
        # Prikažemo izvirno sliko
        print("Original:\n {}".format(slika))
        print("Konvolucija:\n {}".format(slika_konvolucija))
        print("######################")
    

