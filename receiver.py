import sys

from scipy.fft import fft

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt


def fft_symbols(signal):
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

    return np.mod(np.round((np.arctan2(values.imag, values.real) + np.pi) / np.pi * 2 - 2), 4).astype('byte')


overflowcount = 0


def stream_read_left_float32(stream, amount):
    global overflowcount

    # read samples
    window, overflow = stream.read(amount)
    if overflow:
        overflowcount += 1
        print(f'overflow number {overflowcount}')
    if overflowcount > 10:
        print("OVERFLOWED ALL IS LOST")
        exit(1)

    # keep only left channel and convert to float32
    return (window[:, 0] / 2147483647).astype('float32')


def determine_snr(stream):
    # record two seconds for noise
    noise_measurement = stream_read_left_float32(stream, 48000 * 2)
    # return loudest value
    return np.amax(np.abs(noise_measurement))


def symbols_to_bytes(symbols):
    out = np.empty(len(symbols) // 4, dtype='byte')
    for i in range(len(out)):
        out[i] = symbols[i * 4 + 0] << 6
        out[i] |= symbols[i * 4 + 1] << 4
        out[i] |= symbols[i * 4 + 2] << 2
        out[i] |= symbols[i * 4 + 3] << 0
    return out.tobytes()


def main():
    name = 'Loopback: PCM (hw:2,1)'

    stream = sd.InputStream(device=name, samplerate=48000, channels=2, dtype='int32')
    stream.start()

    print('discovering noise floor')
    noise_floor = determine_snr(stream)
    print(f'noise floor at {noise_floor}')

    # twenty seconds of buffer
    data = np.empty(48000 * 20, dtype='float32')
    record_index = 0

    # record half seconds at a time waiting for broken silence
    silence_detector_chunk_size = 48000 // 10
    candidate = np.zeros(silence_detector_chunk_size)

    print("waiting for sound")

    # while no samples exceed noise floor by 10%
    while not np.any(np.abs(candidate) > (noise_floor * snr_factor)):
        candidate = stream_read_left_float32(stream, silence_detector_chunk_size)

    print("sound detected")

    # record while sound exceeds noise floor
    while np.any(np.abs(candidate) > (noise_floor * snr_factor)):
        if record_index * silence_detector_chunk_size >= len(data):
            print("sound too long!")
            break

        dst_index = record_index * silence_detector_chunk_size
        data[dst_index:dst_index + silence_detector_chunk_size] = candidate
        record_index += 1
        candidate = stream_read_left_float32(stream, silence_detector_chunk_size)

    print("sound stopped!")

    stream.stop()
    stream.close()

    angles = fft_symbols(data // symbol_length_samples * symbol_length_samples)
    change = angles[1:] != angles[:-1]
    changes = change.nonzero()[0]
    change_lens = changes[1:] - np.concatenate(([0], changes[:-2]))

    start_sample = 0
    skip_candidates = (change_lens > 7).nonzero()[0]
    for i, v in enumerate(skip_candidates):
        if i != v:
            if i - 1 > 0:
                start_sample = (changes[i - 1] - 2) * symbol_length_samples
                break

    data = data[start_sample:]

    preamble_start = 0
    for i in range(len(data)):
        angles = fft_symbols(data[i:i + preamble_num_samples * 2])
        if ''.join(str(s) for s in preamble) in ''.join(str(a) for a in angles[:len(preamble)]):
            preamble_start = i
            break

    plt.rcParams['figure.dpi'] = 300

    plt.plot(data[preamble_start - symbol_length_samples * 2:preamble_start + symbol_length_samples * 15])
    plt.axvline(symbol_length_samples * 2)
    plt.show()

    data_start = preamble_start + preamble_num_samples

    symbols = fft_symbols(
        data[data_start: data_start + (len(data) - data_start) // symbol_length_samples * symbol_length_samples])

    print(symbols_to_bytes(symbols).decode('ascii'))


if __name__ == '__main__':
    symbol_length_samples = 50
    cycles_per_symbol = 1

    preamble = [0, 2, 1, 3, 0, 0, 1, 1, 2, 2, 3, 3]
    preamble_num_samples = symbol_length_samples * len(preamble)

    # 10% over noise
    snr_desired = 10
    snr_factor = 1. + (snr_desired / 100.)

    main()
