import sys
from time import sleep

from scipy.fft import fft

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt


def bytes_to_symbols(data):
    symbols = np.empty(len(data) * 4, dtype='byte')
    for i, v in enumerate(data):
        symbols[i * 4 + 0] = (v >> 6) & 0b11
        symbols[i * 4 + 1] = (v >> 4) & 0b11
        symbols[i * 4 + 2] = (v >> 2) & 0b11
        symbols[i * 4 + 3] = (v >> 0) & 0b11
    return symbols


def fft_symbols(signal):
    plt.plot(signal)
    plt.show()
    # split signal into sections of symbols
    signal_sections = np.split(signal, len(signal) // symbol_length_samples)
    # perform transformations
    transformations = fft(signal_sections, norm='ortho')

    # take positive frequencies
    transformations = transformations[:, 1:len(transformations[0]) // 2]

    # find the loudest frequency indices
    loud_indices = np.argmax(np.abs(transformations), axis=1)

    # select those values from transformations
    values = np.take_along_axis(transformations, loud_indices.reshape(len(transformations), 1), axis=1).flatten()
    plt.scatter(values.real, values.imag)
    plt.xlim(-2,2)
    plt.ylim(-2,2)
    plt.show()

    return np.mod(np.round((np.arctan2(values.imag, values.real) + np.pi) / np.pi * 2 - 2), 4).astype('byte')

def angle(x,y):
    return np.mod(np.round((np.arctan2(y,x) + np.pi) / np.pi * 2 - 2), 4)


def main():
    sample_space = np.linspace(0, cycles_per_symbol, symbol_length_samples)
    phases = [0, 0.5, 1, 1.5]
    symbol_signals = []
    for i in phases:
        symbol_signals.append((np.cos(2 * np.pi * sample_space + i * np.pi) * 0.2 * 2147483647).astype('int32'))

    symbol_signals = [np.hstack((i.reshape(len(i), 1), np.zeros((len(sample_space), 1), dtype='int32'))) for i in
                      symbol_signals]

    print(angle(1,0))
    print(angle(0,1))
    print(angle(-1,0.1))
    print(angle(0,-1))
    print(angle(-1,-0.1))
    print(angle(0,-1))
    print(fft_symbols(symbol_signals[0][:,0] / 2147483647.))
    print(fft_symbols(symbol_signals[1][:,0] / 2147483647.))
    print(fft_symbols(symbol_signals[2][:,0] / 2147483647.))
    print(fft_symbols(symbol_signals[3][:,0] / 2147483647.))
    return

    with open(sys.argv[1], "rb") as f:
        filedata = f.read()

    symbols = bytes_to_symbols(test_phrase)

    signal = np.concatenate([symbol_signals[i].copy() for i in symbols])

    name = 'Loopback: PCM (hw:2,0)'
    test = sd.query_devices(device=name, kind='output')
    stream = sd.OutputStream(samplerate=48000, device=name, channels=2, dtype='int32')
    # stream = sd.OutputStream(samplerate=48000, device=sd.default.device, channels=2, dtype='int32')

    stream.start()
    # send preamble
    for i in preamble:
        stream.write(symbol_signals[i])

    sleep(1)

    # send data
    # for i in symbols:
        # stream.write(symbol_signals[i])
    stream.stop()
    stream.close()


if __name__ == '__main__':
    test_phrase = b'this is a test of QAM. I really really hope it works out well!'

    preamble = [0, 2, 1, 3, 0, 0, 1, 1, 2, 2, 3, 3]

    symbol_length_samples = 250
    cycles_per_symbol = 1

    main()
