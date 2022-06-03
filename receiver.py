import sys

import scipy.fft
from scipy.fft import fft

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt


def fft_symbols(signal, get_err=False):
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

    non_round = np.mod((np.arctan2(values.imag, values.real) + np.pi) / np.pi * 2 - 2, 4)
    round = np.mod(np.round((np.arctan2(values.imag, values.real) + np.pi) / np.pi * 2 - 2), 4).astype('byte')

    out = np.mod(np.round((np.arctan2(values.imag, values.real) + np.pi) / np.pi * 2 - 2), 4).astype('byte')


    N = len(signal) // (len(signal) // symbol_length_samples)

    scale = scipy.fft.fftfreq(N, 1/sample_rate)[1:N//2]

    try:
        plt.plot(scale, np.abs(transformations[3]))
        plt.show()
    except Exception as e:
        print (e)

    print (f'f Most common loud frequency: {scale[np.bincount(loud_indices).argmax()]}')


    error = out - non_round
    error = np.where(error < -3, 4 + error, error)
    if get_err:
        return out, error
    else:
        return out


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
    # return 0
    # return 0.04766740344464779
    # record four seconds for noise and drop first two.
    noise_measurement = stream_read_left_float32(stream, sample_rate * 4)
    noise_measurement = noise_measurement[sample_rate * 2:]

    # plt.plot(noise_measurement)
    # plt.show()
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

    # stream = sd.InputStream(device=sd.default.device, samplerate=sample_rate, channels=2, dtype='int32')
    stream = sd.InputStream(device=name, samplerate=sample_rate, channels=2, dtype='int32')
    stream.start()

    print('discovering noise floor')
    noise_floor = determine_snr(stream)
    print(f'noise floor at {noise_floor}')

    while True:
        # twenty seconds of buffer
        data = np.empty(sample_rate * 20, dtype='float32')
        record_index = 0

        # record half seconds at a time waiting for broken silence
        silence_detector_chunk_size = sample_rate // 10
        rec_chunk = np.zeros(silence_detector_chunk_size)

        print("waiting for sound")

        # while no samples exceed noise floor by 10%
        while not np.any(np.abs(rec_chunk) > (noise_floor * snr_factor)):
            rec_chunk = stream_read_left_float32(stream, silence_detector_chunk_size)

        print("sound detected")

        # record while sound exceeds noise floor
        while np.any(np.abs(rec_chunk) > (noise_floor * snr_factor)):
            if record_index * silence_detector_chunk_size >= len(data):
                print("sound too long!")
                break

            dst_index = record_index * silence_detector_chunk_size
            data[dst_index:dst_index + silence_detector_chunk_size] = rec_chunk
            record_index += 1
            rec_chunk = stream_read_left_float32(stream, silence_detector_chunk_size)

        print("sound stopped!")
        stop_time = record_index * silence_detector_chunk_size

        angles = fft_symbols(data[:len(data) // symbol_length_samples * symbol_length_samples])
        change = angles[1:] != angles[:-1]
        changes = change.nonzero()[0]
        change_lens = changes[1:] - np.concatenate(([0], changes[1:-1]))

        start_sample = 0
        skip_candidates = (change_lens > 15).nonzero()[0]
        if not len(skip_candidates):
            print("listening for next sound because no preamble found")
            continue

        # check lapses
        for i in range(1, len(skip_candidates)):
            # lapse in skip candidates indicate data
            if skip_candidates[i] - 1 != skip_candidates[i - 1]:
                start_sample = (changes[skip_candidates[i - 1]] - 2) * symbol_length_samples
                break
        # no lapse found, pick last skip
        if start_sample == 0 and len(skip_candidates):
            start_sample = (changes[skip_candidates[-1]+1] - 2) * symbol_length_samples

        if start_sample < 0:
            start_sample = 0

        data = data[start_sample:]
        stop_time -= start_sample

        preamble_start = 0
        for i in range(len(data)):
            if i == 0:
                plt.plot( data[i:i + preamble_num_samples * 2] )
                plt.title('searching preamble in this initial window')
                plt.show()

            angles, error = fft_symbols(data[i:i + preamble_num_samples * 2], True)
            if ''.join(str(s) for s in preamble) in ''.join(str(a) for a in angles[:len(preamble)]):
                preamble_start = i
                break

        error = np.sum(np.abs(error))
        last_error = error
        extra_offset = 0
        while last_error >= error:
            last_error = error
            preamble_error_search_start = preamble_start + extra_offset
            search_angles, error = fft_symbols(
                data[preamble_error_search_start:preamble_error_search_start + preamble_num_samples * 2], True)
            error = np.sum(np.abs(error))
            extra_offset += 1

        # Add the offset excluding the one that increased error
        preamble_start += extra_offset - 1


        plt.plot(data[max(preamble_start - symbol_length_samples * 2, 0):preamble_start + symbol_length_samples * 15])
        plt.title('found preamble')
        if preamble_start - symbol_length_samples * 2 < 0:
            plt.axvline(symbol_length_samples * 2 - (preamble_start - symbol_length_samples * 2))
        else:
            plt.axvline(symbol_length_samples * 2)

        plt.show()

        data_start = preamble_start + preamble_num_samples

        symbols, error = fft_symbols(
            data[data_start: data_start + (stop_time - data_start) // symbol_length_samples * symbol_length_samples],
            True)

        plt.figure(figsize=(100, 5))
        plt.plot(data[data_start:  data_start + (stop_time - data_start) // symbol_length_samples * symbol_length_samples], linewidth=0.5)
        plt.title('data signal')
        plt.show()
        plt.plot(error)
        plt.title('error over time')
        plt.show()

        print(symbols_to_bytes(symbols).decode(encoding='ascii', errors='replace'))

        break

    stream.stop()
    stream.close()


if __name__ == '__main__':
    plt.rcParams['figure.dpi'] = 300

    symbol_length_samples = 100

    sample_rate = 44100

    preamble = [0, 2, 1, 3, 0, 0, 1, 1, 2, 2, 3, 3]
    preamble_num_samples = symbol_length_samples * len(preamble)

    # 10% over noise
    snr_desired = 10
    snr_factor = 1. + (snr_desired / 100.)

    main()
