import cv2 as cv
import numpy as np

def zgradi_filter_nizko_sito(velikost):
    '''Zgradi filter določene velikosti in tipa.'''
    return np.ones((velikost,velikost)) * (1/(velikost**2))

def filtriraj_n_krat_isti_filter(slika, velikost_jedra=3,iteracije=10):
    '''Filtriraj sliko n-krat z istim filtrom.'''
    jedro = zgradi_filter_nizko_sito(3)
    
    for i in range(iteracije):
        slika = cv.filter2D(slika, -1, jedro)
        cv.imshow("Slika filtrirana {}x".format(i+1), slika)
        cv.waitKey(0)
        cv.destroyAllWindows()

def filtriraj_n_krat_vecaj_filter(slika,iteracije=10):
    
    for i in range(1,iteracije):
        jedro = zgradi_filter_nizko_sito(2*i+1)
        slika_filt = cv.filter2D(slika, -1, jedro)
        cv.imshow("Slika filtrirana s filtrom velikosti({}x{})".format(2*i+1,2*i+1), slika_filt)
        cv.waitKey(0)
        cv.destroyAllWindows()

def gaussov_filter(slika, velikost_jedra,sigma):
    '''Filtriraj sliko z Gaussovim filtrom.'''
    return cv.GaussianBlur(slika, (velikost_jedra,velikost_jedra), sigma)


if __name__ == "__main__":
    slika = cv.imread("../.utils/lenna.png")

    filtriraj_n_krat_isti_filter(slika)
    filtriraj_n_krat_vecaj_filter(slika)   

    #Gaussov filter
    gauss_filt = gaussov_filter(slika,15,0)
    cv.imshow("Gaussov filter", gauss_filt)
    cv.waitKey(0)