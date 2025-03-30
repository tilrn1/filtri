import cv2 as cv
import numpy as np

def dodaj_sum(slika,odstotek):
    '''Dodaj šum sliki.'''
    slika_s_sumom = slika.copy()
    st_pikslov = int(slika.shape[0] * slika.shape[1]*odstotek)

    for i in range(st_pikslov):
        x = np.random.randint(0, slika.shape[1])
        y = np.random.randint(0, slika.shape[0])
        slika_s_sumom[y,x] = [255,255,255]
    return slika_s_sumom

def median_filter(slika, velikost_jedra):
    '''Filtriraj sliko z median filtrom.'''
    return cv.medianBlur(slika, velikost_jedra)

if __name__ == "__main__":

    slika = cv.imread("../.utils/lenna.png")
    #Dodajmo sliki šum
    slika_s_sumom_1 = dodaj_sum(slika.copy(),0.1)
    slika_s_sumom_2 = dodaj_sum(slika.copy(),0.2)
    slika_s_sumom_3 = dodaj_sum(slika.copy(),0.3)
    slika_s_sumom_4 = dodaj_sum(slika.copy(),0.4)

    #Median filter
    velikost_median_filtra = 3
    for i in range(1,5):
        cv.imshow("Slika s sumom {}".format(i), eval("slika_s_sumom_{}".format(i)))
        median_filt = median_filter(eval("slika_s_sumom_{}".format(i)),velikost_median_filtra)
        cv.imshow("Median filter slika_{}".format(i), median_filt)
        cv.waitKey(0)
    cv.destroyAllWindows()