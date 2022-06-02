import sys

from scipy.fft import fft

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt


def main():
    symbol_length_samples = 50
    cycles_per_symbol = 0.5

    name = 'Loopback: PCM (hw:2,1)'
    test = sd.query_devices(device=name, kind='input')

    stream = sd.InputStream(device=name, samplerate=48000, channels=2, dtype='int32')
    stream.start()

    data, overflow = stream.read(1024)

    while np.amax(data) == 0:
        data, overflow = stream.read(symbol_length_samples)




    stream.stop()
    stream.close()

    # plt.rcParams['figure.dpi'] = 300
    #
    # signal_sections = np.split(signal, len(signal) // symbol_length_samples)
    # transformations = fft(signal_sections, norm='ortho')
    #
    # # take positive frequencies
    # transformations = transformations[:, 1:len(transformations[0]) // 2]
    #
    # # find the loudest frequency indices
    # loud_indices = np.argmax(np.abs(transformations), axis=1)
    #
    # # select those values from transformations
    # values = np.take_along_axis(transformations,loud_indices.reshape(len(transformations),1) ,axis=1).flatten()
    #
    # # plot the stuff
    # plt.scatter(values.real, values.imag, 10, cmap=plt.cm.rainbow, c=np.linspace(0, 1, len(values)))
    # plt.xlim(-1, 1)
    # plt.ylim(-1, 1)
    # plt.show()
    # plt.cla()


if __name__ == '__main__':
    main()
