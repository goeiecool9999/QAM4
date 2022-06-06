from scipy.fft import fft

import sounddevice as sd
import numpy as np


def fft_symbols_error_feedback(signal):
    """
    Scans a signal for symbols in chunks, while using the error output to adjust the next analysis.
    Thus keeping the decoding in phase.
    :param signal: The signal to be processed
    :return: numpy array of decoded symbols
    """
    symbol_chunk_size = 150

    symbols = []
    chunk_start = 0
    chunk_end = symbol_chunk_size * symbol_length_samples

    while chunk_end < len(signal):
        angles, error = fft_symbols(signal[chunk_start: chunk_end], True)
        symbols.append(angles)

        # 0.5 error for a symbol means 90 degrees phase shift
        # 90 degrees shift = 0.25*symbol_length_samples
        correction = int(np.sum(error) / symbol_chunk_size * 0.25 * symbol_length_samples)

        chunk_start = chunk_end + correction
        chunk_end = chunk_start + symbol_chunk_size * symbol_length_samples

    return np.concatenate(symbols)


def fft_symbols(signal, get_err=False):
    """
    Takes a signal and naively decodes it with a series of FFT's.
    :param signal: signal to be analysed
    :param get_err: when true also return error values
    :return: if get_err = True returns tuple of symbols and corresponding error values, otherwise returns symbols
    """
    # split signal into sections of symbols
    signal_sections = np.split(signal, len(signal) // symbol_length_samples)

    # perform transformations
    transformations = fft(signal_sections, norm='ortho')

    # take target frequency
    values = transformations[:,cycles_per_symbol]

    # rounded angle calculations becoming symbol values
    out = np.mod(np.round((np.arctan2(values.imag, values.real) + np.pi) / np.pi * 2 - 2), 4).astype('byte')

    if get_err:
        # unrounded angle calculations to get error
        raw = np.mod((np.arctan2(values.imag, values.real) + np.pi) / np.pi * 2 - 2, 4)

        # calculate error
        error = out - raw
        error = np.where(error < -3, 4 + error, error)
        return out, error
    else:
        return out


def stream_read_left_float32(stream, amount):
    """
    Read specified amount of float32 samples from streams left channel
    :param stream: stream to read from
    :param amount: amount of samples to read
    :return: numpy float32 array of samples
    """
    # read samples
    window, overflow = stream.read(amount)

    # signal is useless when samples are skipped. terminate.
    if overflow:
        print("OVERFLOWED ALL IS LOST")
        exit(1)

    # keep only left channel and convert to float32
    return (window[:, 0] / 2147483647).astype('float32')


def determine_snr(stream):
    # record four seconds for noise and drop first two.
    noise_measurement = stream_read_left_float32(stream, sample_rate * 4)
    noise_measurement = noise_measurement[sample_rate * 2:]

    # return loudest value
    return np.amax(np.abs(noise_measurement))


def symbols_to_bytes(symbols):
    """
    Convert signal symbols back to bytes.
    :param symbols: list of symbols
    :return: numpy array of bytes
    """
    out = np.empty(len(symbols) // 4, dtype='byte')
    for i in range(len(out)):
        out[i] = symbols[i * 4 + 0] << 6
        out[i] |= symbols[i * 4 + 1] << 4
        out[i] |= symbols[i * 4 + 2] << 2
        out[i] |= symbols[i * 4 + 3] << 0
    return out.tobytes()


def main():
    name = 'Loopback: PCM (hw:2,1)'

    stream = sd.InputStream(device=sd.default.device, samplerate=sample_rate, channels=2, dtype='int32')
    # stream = sd.InputStream(device=name, samplerate=sample_rate, channels=2, dtype='int32')
    stream.start()

    print('discovering noise floor')
    noise_floor = determine_snr(stream)
    print(f'noise floor at {noise_floor}')

    while True:
        # twenty seconds of buffer
        data = np.empty(sample_rate * 20, dtype='float32')
        record_index = 0

        # record 10th of a second at a time waiting for significant sound
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

        # !!!!!!!! HEY PAY ATTENTION THIS IS IMPORTANT YOU WILL FORGET ABOUT THIS !!!!!!!!
        # Some sound cards invert the signal which trips up the preamble detection logic.
        # Comment or uncomment when appropriate
        # TODO: Automatically detect inverted signal
        data = -data

        # Detect leading constant wave that precedes the preamble pattern.
        # Naive fft which is good enough to catch a bunch of unchanging phases.
        angles = fft_symbols(data[:stop_time // symbol_length_samples * symbol_length_samples])
        # compare each element to the previous element to know if there was a change
        change = angles[1:] != angles[:-1]
        # Make a list of all the indices where the symbol was detected to change
        changes = change.nonzero()[0]
        # Find the lengths of stretches of unchanging phase.
        change_lens = changes - np.concatenate(([0], changes[0:-1]))

        # Consider only stretches of more than 15 unchanged phases.
        skip_candidates = (change_lens > 15).nonzero()[0]
        if not len(skip_candidates):
            print("listening for next sound because no preamble found")
            continue

        start_sample = 0

        # check lapses
        for i in range(1, len(skip_candidates)):
            # lapse in skip candidates indicate data
            if skip_candidates[i] - 1 != skip_candidates[i - 1]:
                start_sample = (changes[skip_candidates[i - 1]]) * symbol_length_samples
                break

        # no lapse found, pick last skip
        if start_sample == 0 and len(skip_candidates):
            start_sample = (changes[skip_candidates[-1]]) * symbol_length_samples

        # Discard all the data that comes before the end of the stretch
        data = data[start_sample:]
        # Update stop time to reflect that.
        stop_time -= start_sample

        # Find the actual preamble pattern to recognise.
        preamble_start = 0
        missing_preamble = True
        for i in range(preamble_num_samples * 2):
            angles, error = fft_symbols(data[i:i + preamble_num_samples * 2], True)
            # TODO: Find a better way to test for non string sequences
            # converts the decoded symbols and preamble to strings and checks
            # if the string of the preamble is in the string of the decoded symbols
            if ''.join(str(s) for s in preamble) in ''.join(str(a) for a in angles[:len(preamble)]):
                preamble_start = i
                missing_preamble = False
                break

        if missing_preamble:
            print('failed to find preamble in sound')
            continue

        # At this point it's likely that the preamble was detected even though
        # the fourier analysis window is not aligned to the symbols perfectly
        # scan forwards slowly until we find a minimum in the summed error values

        error = np.sum(np.abs(error))
        last_error = error
        extra_offset = 0
        while last_error >= error:
            last_error = error
            alignment_search_start = preamble_start + extra_offset
            search_angles, error = fft_symbols(
                data[alignment_search_start:alignment_search_start + preamble_num_samples], True)
            error = np.sum(np.abs(error))
            extra_offset += 1

        # Add the found offset excluding the one that increased error
        preamble_start += extra_offset - 1

        data_start = preamble_start + preamble_num_samples

        symbols = fft_symbols_error_feedback(
            data[data_start: data_start + (stop_time - data_start) // symbol_length_samples * symbol_length_samples]
        )

        print(symbols_to_bytes(symbols).decode(encoding='ascii', errors='replace'))

        break

    stream.stop()
    stream.close()


if __name__ == '__main__':
    symbol_length_samples = 15
    cycles_per_symbol = 1

    sample_rate = 192000

    preamble = [3, 2, 1, 0, 3, 1, 2, 0, 3, 3, 0, 0, 1, 1, 3, 3]
    preamble_num_samples = symbol_length_samples * len(preamble)

    # 10% over noise
    snr_desired = 30
    snr_factor = 1. + (snr_desired / 100.)

    main()
