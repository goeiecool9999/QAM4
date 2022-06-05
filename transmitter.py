import sys
from time import sleep

import sounddevice as sd
import numpy as np


def bytes_to_symbols(data):
    symbols = np.empty(len(data) * 4, dtype='byte')
    for i, v in enumerate(data):
        symbols[i * 4 + 0] = (v >> 6) & 0b11
        symbols[i * 4 + 1] = (v >> 4) & 0b11
        symbols[i * 4 + 2] = (v >> 2) & 0b11
        symbols[i * 4 + 3] = (v >> 0) & 0b11
    return symbols


def main():
    # input for sine
    sample_space = np.linspace(0, cycles_per_symbol, symbol_length_samples)
    # shifts for each symbol
    phases = [0, 0.5, 1, 1.5]
    # sample date for each shifted wave
    symbol_signals = []
    for i in phases:
        # generate single channel sample data as 32-bit integers
        symbol_signals.append((np.cos(2 * np.pi * sample_space + i * np.pi) * 0.6 * 2147483647).astype('int32'))

    # make stereo with silent right channel
    symbol_signals = [np.hstack((i.reshape(len(i), 1), np.zeros((len(sample_space), 1), dtype='int32'))) for i in
                      symbol_signals]

    # read file from argument
    with open(sys.argv[1], "rb") as f:
        filedata = f.read()

    # convert to symbols
    symbols = bytes_to_symbols(filedata)

    name = 'Loopback: PCM (hw:2,0)'
    name = 'HDMI: 3 (hw:0'

    # stream = sd.OutputStream(samplerate=sample_rate, device=name, channels=2, dtype='int32')
    stream = sd.OutputStream(samplerate=sample_rate, device=sd.default.device, channels=2, dtype='int32')

    # start playing the signal
    stream.start()

    # send preamble
    for i in [0] * 50 + preamble:
        stream.write(symbol_signals[i])

    # send data
    for i in symbols:
        stream.write(symbol_signals[i])

    # exit
    sleep(1)
    stream.stop()
    stream.close()


if __name__ == '__main__':
    # Some global parameters
    test_phrase = b'this is a test of QAM. I really really hope it works out well!'

    preamble = [3, 2, 1, 0, 3, 1, 2, 0, 3, 3, 0, 0, 1, 1, 3, 3]

    sample_rate = 192000

    symbol_length_samples = 15
    cycles_per_symbol = 1

    main()
