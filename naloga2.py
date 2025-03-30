import cv2 as cv
import numpy as np
import math

def konvolucija(slika, jedro):
    '''Izvede konvolucijo nad sliko. Brez uporabe funkcije cv.filter2D, ali katerekoli druge funkcije, ki izvaja konvolucijo.
    Funkcijo implementirajte sami z uporabo zank oz. vektorskega računanja.'''
    visina, sirina = slika.shape
    j_visina, j_sirina = jedro.shape
    pad_v = j_visina // 2
    pad_s = j_sirina // 2
      
    razsirjena_slika = np.pad(slika, ((pad_v, pad_v), (pad_s, pad_s)))
    filtrirano = np.zeros_like(slika, dtype=np.float32)
    for i in range(visina):
        for j in range(sirina):
            filtrirano[i, j] = np.sum(razsirjena_slika[i:i+j_visina, j:j+j_sirina] * jedro)
    
    return filtrirano
    pass

def filtriraj_z_gaussovim_jedrom(slika,sigma):
    '''Filtrira sliko z Gaussovim jedrom..'''
    velikost_jedra = (int)(2 * sigma) * 2 + 1
    k = (velikost_jedra / 2) - (1/2)
  
    jedro = np.zeros((velikost_jedra, velikost_jedra), dtype=np.float32)
    for i in range(velikost_jedra):
        for j in range(velikost_jedra):
            jedro[i,j] = (1 / (2 * math.pi * math.pow(sigma, 2)) * math.exp(-(math.pow((i - k - 1), 2) + math.pow((j - k - 1), 2)) / (2 * math.pow(sigma, 2))))
  
    jedro /= np.sum(jedro)
    return konvolucija(slika,jedro)
    pass

def filtriraj_sobel_smer(slika):
    '''Filtrira sliko z Sobelovim jedrom in označi gradiente v orignalni sliki glede na ustrezen pogoj.'''
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    gradient_y = konvolucija(slika, sobel_y)
    barvna_slika = cv.cvtColor(slika.astype(np.uint8), cv.COLOR_GRAY2BGR)
 
    barvna_slika[np.where(gradient_y > 150)] = [0, 255, 0]
    return barvna_slika
    pass

if __name__ == '__main__':   
    slika = cv.imread('.utils/wolf.jpg', cv.IMREAD_GRAYSCALE).astype(np.float32)
    if slika is None:
        print("Napaka: Slika ni bila naložena. Preveri pot do slike.")
    else:
        gaus_slika = filtriraj_z_gaussovim_jedrom(slika, 1) 
        sobel_vertikalna = filtriraj_sobel_smer(gaus_slika)
         
        while True:  
            cv.imshow('Slika', slika.astype(np.uint8))
            cv.imshow('Gauss filter', gaus_slika.astype(np.uint8))
            cv.imshow('Sobel filter', sobel_vertikalna)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
  
    cv.waitKey(0)
    cv.destroyAllWindows()
    pass
