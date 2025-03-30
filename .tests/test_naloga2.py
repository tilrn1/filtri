import sys
import os
sys.path.append(os.getcwd())

import naloga2
import numpy as np
import os,sys
import importlib
import inspect
import pytest

def test_konvolucija():
    slika_imp = np.zeros((5, 5, 5), dtype=np.float32)
    slika_imp[2, 2, 0] = 1
    slika_imp[0, 0, 1] = 1
    slika_imp[0, -1, 2] = 1
    slika_imp[-1, -1, 3] = 1
    slika_imp[-1, 0, 4] = 1

    jedro = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)

    slika_filt = np.zeros((5, 5, 5), dtype=np.float32)
    slika_filt[:, :, 0] = [
            [0, 0, 0, 0, 0],
            [0, 9, 8, 7, 0],
            [0, 6, 5, 4, 0],
            [0, 3, 2, 1, 0],
            [0, 0, 0, 0, 0],
            ]
    slika_filt[:, :, 1] = [
            [5, 4, 0, 0, 0],
            [2, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            ]
    slika_filt[:, :, 2] = [
            [0, 0, 0, 6, 5],
            [0, 0, 0, 3, 2],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            ]
    slika_filt[:, :, 3] = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 9, 8],
            [0, 0, 0, 6, 5],
            ]
    slika_filt[:, :, 4] = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [8, 7, 0, 0, 0],
            [5, 4, 0, 0, 0],
            ]

    for n in range(5):
        i = slika_imp[:, :, n].copy()
        j = jedro.copy()
        r = naloga2.konvolucija(i, j)

        np.testing.assert_array_equal(slika_imp[:, :, n], i, err_msg='Podana vhodna slika se je spremenila. Slike ne smete spreminjati!')
        np.testing.assert_array_equal(jedro, j, err_msg='Podano jedro se je spremenilo. Vhodnega jedra ne smete spreminjati!')

        np.testing.assert_array_equal(r, slika_filt[:, :, n], err_msg=f'Rezultat konvolucije se ne ujema z pričakovanim, korak {n}.', verbose=True)


def test_filtriraj_z_gaussovim_jedrom():

    slika_imp = np.zeros((15, 15), dtype=np.float32)
    slika_imp[7, 7] = 1

    i = slika_imp.copy()
    slika_filt = naloga2.filtriraj_z_gaussovim_jedrom(i, 2.)
    np.testing.assert_array_equal(slika_imp, i, err_msg='Podana vhodna slika se je spremenila. Slike ne smete spreminjati!')

    ref_vect = np.array([0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0], np.float32)
    ref_vect = np.array([0.      , 0.      , 0.      , 0.005385, 0.012918, 0.024133,
                            0.035113, 0.039789, 0.035113, 0.024133, 0.012918, 0.005385,
                            0.      , 0.      , 0.      ])

    np.testing.assert_allclose(slika_filt[7, :], ref_vect, rtol=1e-3, verbose=True)
    np.testing.assert_allclose(slika_filt[:, 7], ref_vect, rtol=1e-3, verbose=True)

    ref_vect = np.array([0.      , 0.002393, 0.00441 , 0.00727 , 0.010726, 0.01416 ,
                            0.016728, 0.017684, 0.016728, 0.01416 , 0.010726, 0.00727 ,
                            0.00441 , 0.002393, 0.      ])

    slika_filt = naloga2.filtriraj_z_gaussovim_jedrom(i, 3.)

    np.testing.assert_allclose(slika_filt[7, :], ref_vect, rtol=1e-3, verbose=True)
    np.testing.assert_allclose(slika_filt[:, 7], ref_vect, rtol=1e-3, verbose=True)

def test_filtriraj_sobel():
    slika_imp = np.zeros((7, 7), dtype=np.float32)
    slika_imp[3, :] = 1
    slika_imp[:, 3] = 1

    vertikalno_obstaja = does_function_exist("naloga2", "filtriraj_sobel_vertikalno") 
    horizontalno_obstaja = does_function_exist("naloga2", "filtriraj_sobel_horizontalno")
    if vertikalno_obstaja and horizontalno_obstaja:
        print("Implementirati morate samo eno od funkcij filtriraj_sobel_vertikalno oz. filtriraj_sobel_horizontalno. Poglejte v navodila.")
        assert False
    elif vertikalno_obstaja:
        slika_filt = naloga2.filtriraj_sobel_vertikalno(slika_imp)
        ref_vect1 = np.array([4, 4, 3, 2, 3, 4, 4])
        ref_vect2 = np.array([-4, -4, -3, -2, -3, -4, -4])       
        np.testing.assert_allclose(slika_filt[:, 2], ref_vect1, rtol=1e-3, verbose=True)
        np.testing.assert_allclose(slika_filt[:, 4], ref_vect2, rtol=1e-3, verbose=True)
        
    elif horizontalno_obstaja:
        slika_filt = naloga2.filtriraj_sobel_horizontalno(slika_imp)
        
        ref_vect1 = np.array([4, 4, 3, 2, 3, 4, 4])
        ref_vect2 = np.array([-4, -4, -3, -2, -3, -4, -4])
        np.testing.assert_allclose(slika_filt[2, :], ref_vect1, rtol=1e-3, verbose=True)
        np.testing.assert_allclose(slika_filt[4, :], ref_vect2, rtol=1e-3, verbose=True)


@pytest.mark.skip
def does_function_exist(module_name, function_name):
    try:
        # Import the module dynamically
        module = importlib.import_module(module_name)

        # Get the function object
        function_obj = getattr(module, function_name)

        # Check if it's a callable function
        return inspect.isfunction(function_obj)
    except (ImportError, AttributeError):
        return False

