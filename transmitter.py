import sys

from scipy.fft import fft

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt

test_phrase = b'this is a test of QAM. I really really hope it works out well!'


def bytes_to_symbols(data):
    symbols = np.empty(len(data) * 4, dtype='byte')
    for i, v in enumerate(data):
        symbols[i * 4 + 0] = (v >> 6) & 0b11
        symbols[i * 4 + 1] = (v >> 4) & 0b11
        symbols[i * 4 + 2] = (v >> 2) & 0b11
        symbols[i * 4 + 3] = (v >> 0) & 0b11
    return symbols

def main():
    symbol_length_samples = 50
    cycles_per_symbol = 0.5

    sample_space = np.linspace(0, cycles_per_symbol, symbol_length_samples)
    phases = [0, 0.5, 1, 1.5]
    symbol_signals = []
    for i in phases:
        symbol_signals.append((np.cos(2 * np.pi * sample_space + i * np.pi) * 0.2 * 2147483647).astype('int32'))

    symbol_signals = [np.hstack((i.reshape(len(i),1), np.zeros( (len(sample_space), 1) , dtype='int32') )) for i in symbol_signals]

    with open(sys.argv[1], "rb") as f:
        filedata = f.read()

    symbols = bytes_to_symbols(test_phrase)

    signal = np.concatenate([symbol_signals[i].copy() for i in symbols])

    name = 'Loopback: PCM (hw:2,0)'
    test = sd.query_devices(device=name, kind='output')
    stream = sd.OutputStream(samplerate=48000, device=name, channels=2, dtype='int32')

    stream.start()
    for i in symbols:
        stream.write(symbol_signals[i])
    stream.stop()
    stream.close()


if __name__ == '__main__':
    main()
