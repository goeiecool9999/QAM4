from math import atan
import sys
from time import sleep

import sounddevice as sd
import numpy as np
from matplotlib import pyplot as plt


def bytes_to_symbols(data):
    symbols = np.empty(len(data) * 2, dtype='byte')
    for i, v in enumerate(data):
        symbols[i * 2 + 0] = (v >> 4) & 0b1111
        symbols[i * 2 + 1] = (v >> 0) & 0b1111
    return symbols


def main():
    # input for sine
    sample_space = np.linspace(0, cycles_per_symbol, symbol_length_samples, endpoint=False)

    # phase amplitude pairs for each symbol
    symbol_wave_parameters = []

    # upper 2 bits determine quadrant of constellation
    # lower 2 bits determine position within quadrant

    # Base phase of each quadrant
    quadrant_base_phases = [0.25, 1.75, 0.75, 1.25]
    # phase offset for each position in quadrant
    quadrant_location_phases = [0, -0.125, 0.125, 0]

    quiet = 0.25
    loud = 0.75

    # amplitude offset for each position in quadrant
    quadrant_location_amplitudes = [quiet, loud, loud, loud]

    for quadrant in range(4):
        for lower_bits in range(4):
            symbol_wave_parameters.append((quadrant_base_phases[quadrant] + quadrant_location_phases[lower_bits],
                                           quadrant_location_amplitudes[lower_bits]))


    # sample date for each shifted wave
    symbol_signals = []
    for phase, amp in symbol_wave_parameters:
        # generate single channel sample data as 32-bit integers
        symbol_signals.append((np.cos(2 * np.pi * sample_space + phase * np.pi) * amp * 2147483647).astype('int32'))

    # fig, axs = plt.subplots(4, 4)
    # for i, v in enumerate(symbol_signals):
    #     axs[i//4,i%4].set_title(f'{i}')
    #     axs[i//4,i%4].plot(v)
    #
    # plt.show()
    # plt.close(fig)

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

    stream = sd.OutputStream(samplerate=sample_rate, device=name, channels=2, dtype='int32')
    # stream = sd.OutputStream(samplerate=sample_rate, device=sd.default.device, channels=2, dtype='int32')

    # start playing the signal
    stream.start()

    # send preamble
    # for i in [x for x in range(16)] * 50 + preamble:
    for i in [3] * 50 + preamble:
        underflowed = stream.write(symbol_signals[i])
        if underflowed:
            print("YOU DONKEY")
            exit(1)

    # send data
    for i in symbols:
        underflowed = stream.write(symbol_signals[i])
        if underflowed:
            print("YOU DONKEY")
            exit(1)

    # exit
    sleep(1)
    stream.stop()
    stream.close()


if __name__ == '__main__':
    # Some global parameters
    test_phrase = b'this is a test of QAM. I really really hope it works out well!'

    preamble = [5, 9, 13, 12, 7, 2, 3, 14, 1, 8, 6, 4, 10, 11, 15, 0]

    sample_rate = 48000

    symbol_length_samples = 20
    cycles_per_symbol = 1

    main()
