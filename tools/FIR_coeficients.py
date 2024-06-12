import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz

def design_fir_filter(filter_order, cutoff_freq, fs, filter_type='bandpass', window='hamming'):
    """
    Diseña un filtro FIR.

    Parámetros:
    - filter_order: Orden del filtro (entero)
    - cutoff_freq: Frecuencia(s) de corte (lista de dos valores para bandpass)
    - fs: Frecuencia de muestreo
    - filter_type: Tipo de filtro ('low', 'high', 'bandpass', 'bandstop')
    - window: Tipo de ventana a utilizar (por defecto 'hamming')

    Retorna:
    - b: Coeficientes del filtro FIR
    """
    if isinstance(cutoff_freq, list):
        normalized_cutoff = [f / (0.5 * fs) for f in cutoff_freq]
    else:
        raise ValueError("Para un filtro pasabanda, 'cutoff_freq' debe ser una lista de dos valores.")
    
    # Diseñar el filtro FIR
    b = firwin(filter_order, normalized_cutoff, window=window, pass_zero=filter_type)
    
    # Calcular la respuesta en frecuencia del filtro
    w, h = freqz(b, worN=8000, fs=fs)
    
    # Graficar la respuesta en frecuencia
    plt.figure()
    plt.plot(w, 20 * np.log10(np.abs(h)), 'b')
    plt.title(f'Respuesta en Frecuencia del Filtro FIR {filter_type} discretizado a {fs} Hz')
    plt.xlabel('Frecuencia (Hz)')
    plt.ylabel('Magnitud (dB)')
    plt.grid()

    # Marcar los puntos de -3 dB
    h_dB = 20 * np.log10(np.abs(h))
    plt.axhline(-3, color='r', linestyle='--')  # Línea en -3 dB

    # Encontrar las frecuencias donde la respuesta en frecuencia cruza -3 dB
    crossing_points = np.where(np.diff(np.sign(h_dB + 3)))[0]
    for cp in crossing_points:
        plt.plot(w[cp], h_dB[cp], 'ro')  # Marcar con un punto rojo

    plt.savefig(f'fir_{cutoff_freq}Hz_{filter_type}_in_{fs}Hz.png')  # Guardar la imagen
    plt.show()
    
    return {'coefs': b, 'num_coef': list(range(len(b)))}

if __name__ == "__main__":
    # Ejemplo de uso
    filter_order = 11
    cutoff_freq = [300]  # Frecuencias de corte para el filtro pasabanda
    fs = 10000                 # Frecuencia de muestreo en Hz
    filter_type = 'lowpass'  # Tipo de filtro
    window = 'hamming'        # Tipo de ventana

    # Diseñar el filtro FIR pasabanda
    resultados = design_fir_filter(filter_order, cutoff_freq, fs, filter_type, window)

    # Imprimir los coeficientes del filtro FIR y su número correspondiente
    for i, coeficiente in zip(resultados['num_coeficientes'], resultados['coeficientes']):
        print(f"Coeficiente {i}: {coeficiente}")
