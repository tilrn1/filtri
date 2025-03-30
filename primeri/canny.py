# Opis: Program, ki odpre kamero in prikazuje originalno sliko, detektirane robove in sliko z efektom.
import cv2
import numpy as np
global spodnji_prag, zgornji_prag
spodnji_prag = 10
zgornji_prag = 100

def beri_spodnji_prag(vrednost):
    global spodnji_prag
    spodnji_prag = vrednost
    
def beri_zgornji_prag(vrednost):
    global zgornji_prag
    zgornji_prag = vrednost

def zaznaj_robove(slika):
    global spodnji_prag, zgornji_prag
    siva = cv2.cvtColor(slika,cv2.COLOR_BGR2GRAY)      
    return cv2.Canny(siva, spodnji_prag, zgornji_prag)

def dodaj_efekt(slika,robovi):
    slika_efekt = cv2.cvtColor(slika,cv2.COLOR_BGR2HSV)
    indeksi = np.where(robovi>0)
    slika_efekt[indeksi[0],indeksi[1],1] = 255                
    return cv2.cvtColor(slika_efekt,cv2.COLOR_HSV2BGR)
    

if __name__ == "__main__":
    img = cv2.imread("lenna.png")
    cv2.namedWindow("Drsnika")
    cv2.createTrackbar("Low threshold","Drsnika",0,255,beri_spodnji_prag)
    cv2.createTrackbar("High threshold","Drsnika",0,255,beri_zgornji_prag)
    kamera = cv2.VideoCapture(0)
    if kamera.isOpened() == False:
        print("Ne morem odpreti kamere")
    else:
        cv2.namedWindow("Kamera-original")
        cv2.namedWindow("Kamera-robovi")
        cv2.namedWindow("Kamera-efekt")
        while True:
            ret, slika = kamera.read()
            slika = cv2.flip(slika,1)
            if ret == True: 
                robovi = zaznaj_robove(slika)
                cv2.imshow("Kamera-original",slika)
                cv2.imshow("Kamera-robovi",robovi)
                cv2.imshow("Kamera-efekt",dodaj_efekt(slika,robovi))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:   
                break
        kamera.release()
        cv2.destroyAllWindows()