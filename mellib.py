import os
from typing import NoReturn
import numpy as np
import scipy.fftpack
import matplotlib.pyplot as plt
from scipy.signal.windows import hamming
from scipy.io.wavfile import read
import soundfile as sf

import matplotlib as mpl
import matplotlib.font_manager as font_manager

# Plot option shenanigans
mpl.rcParams['font.family'] = 'serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif'] = cmfont.get_name()
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['axes.unicode_minus'] = False



WIN_SIZE = 0.025
WIN_STEP = 0.01
NFFT = 512





def plot_signal(signal: list, fs: int) -> NoReturn:
	"""
	Plots the given signal.

	Parameters
	----------
	signal: `list`
		The signal that should be plotted.

	fs: `int`
		The sampling rate of the signal.

	Returns
	-------
	None
	"""

	assert fs > 0, "Frequency sample must be a positive integer."
	assert type(fs) == int, "The sampling rate must be a positive integer."

	plt.plot(signal)
	xticks, xlabel = plt.xticks()
	xticks = np.arange(0, len(signal), fs / 2)  # Ticks avery half a second.
	xlabels = np.arange(0, len(xticks) / 2, 0.5)  # Convert initial unit to seconds.
	if xticks[-1] < len(signal):
		xticks = np.append(xticks, len(signal))
		xlabels = np.append(xlabels, xlabels[-1] + 0.5)
	plt.xticks(xticks, xlabels)
	plt.title(f"Raw signal (sample frequency : {fs})")
	plt.xlabel(f"Time (s) | {len(signal) / fs} seconds | {len(signal)} samples")
	plt.ylabel("Frequency (Hz)")
	plt.grid()
	plt.show()

def wav_to_signal(audio_path: str) -> tuple:
	"""
	Converts a raw audio .wav file into a workable signal.

	Parameters
	----------
	audio_path: `str`
		The path to a given audio.

	Returns
	-------
	signal_data: `tuple`
		Returns a tuple containing the signal and the sample rate.
	"""

	signal, fs = sf.read(audio_path)
	return (signal, fs)


def pre_emphasis(signal, fs, alpha=0.97, visualisation=False) -> list:
	"""
	Accentuates the high frequencies components.

	Parameters
	----------
	signal: `list`
		The signal on which to perform the pre-emphasis.
	fs: `int`
		The frequency at which the signal has been sampled.
	alpha: `float`
		Alpha factor, the ratio by which the signal is going te be pre-emphasised usually set to 0.95 or 0.97
		(default is `0.97`).
	visualisation: `bool`
		Provides a graphical illustration of the operation
		(default is False).

	Returns
	-------
	pre_emphasised_signal: `list`
		The pre-emphasised signal.
	"""

	assert fs > 0, "Sampling rate must be a positive integer."
	assert 0 < alpha <= 1, "The alpha parameter must be a number between strictly greater \
	than 0 and less or equal to 1."  # 1 keeps the signal as it is though...

	pre_emphasised_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])

	if visualisation:
		plt.plot(pre_emphasised_signal)
		xticks, _ = plt.xticks()
		xticks = np.arange(0, len(signal), fs / 2)  # Ticks avery half a second.
		xlabels = np.arange(0, len(xticks) / 2, 0.5)  # Convert initial unit to seconds.
		if xticks[-1] < len(signal):
			xticks = np.append(xticks, len(signal))
			xlabels = np.append(xlabels, xlabels[-1] + 0.5)
		plt.xticks(xticks, xlabels)
		plt.title("Pre-emphasised signal")
		plt.xlabel(f"Time (ms) | {len(signal) / fs} seconds | {len(pre_emphasised_signal)} values")
		plt.ylabel("Emphasised frequency (Hz)")
		plt.grid()
		plt.show()

	return pre_emphasised_signal

def hertz_to_mel(f: list/float/int) -> list/float:
	"""
	Converts a given frequency value to mels.
	
	Parameters
	----------
	f: `float`/`list`
		A frequency or a list of frequencies to convert.

	Returns
	-------
	mel: `float`/`list`
		The converted value of the given frequency, in mels.
	"""

	return 1125 * np.log(1 + (f / 700))

def mel_to_hertz(m: list/float/int) -> list/float:
	"""
	Converts a given mel to frequencies.

	Parameters
	----------
	m: `int`/`float`/`list`
		A mel or a list of mels to convert.
	
	Returns
	-------
	f: `float`/`list`
		The converted value of the given mel, in frequencies.
	"""

	return 700 * (np.exp(m / 1125) - 1)

def frame_signal(
	signal: list,
	fs: int,
	win_size: int=WIN_SIZE,
	win_step: int=WIN_STEP,
	visualisation: bool=False) -> np.ndarray:
	"""
	Frames the given signal according to a window size, a window step and
	its sampling rate.

	Parameters
	----------
	signal: `list`
		The signal to frame.
	fs: `int`
		The sampling rate at which the signal has been recorded.
	win_size: `float`
		The window size, the time period which defines a single frame, in seconds
		(default is `0.0025`).
	win_step: `float`
		The window step, the frequency at which a new frame starts, in seconds
		(default is `0.01`).
	visualisation: `bool`
		Provides a graphical illustration of the operation.
		(default is `False`).

	Returns
	-------
	framed_signal: np.ndarray
		A 2-dimensionnal numpy array representing the framed signal.
	"""

	assert len(signal) > 0, "Signal should be a list which length is more than 0."
	assert fs > 0, "Sampling rate should be an integer geater than 0."
	assert win_size > 0, "Window size must be a float greater than 0."
	assert win_step > 0, "Window step must be a float greater than 0."

	frame_length = int(np.around(win_size * fs))
	frame_increment = int(np.around(win_step * fs))
	n_frames = int(np.ceil((len(signal) - frame_length) / frame_increment))
	complete_samples = n_frames * frame_increment + frame_length

	zero_padding = np.zeros(complete_samples - len(signal))  # Creating the
	# required padding if signal is not long enough for the last frame.
	padded_signal = np.append(signal, zero_padding)

	raw_indices = np.tile(np.arange(0, frame_length), (n_frames, 1))  # Creating a \
	# (n_frames, frame_length) shaped array each row having values from 0 to frame_length - 1
	increment = np.tile(np.arange(0, n_frames * frame_increment, frame_increment), (frame_length, 1)).T # Creating a
	# (n_frames, frame_increment) shaped array filled with values from 0 to n_frames * frame_increment - 1
	# with a frame_increment step. We then transpose it to perform the addition.
	indices = (raw_indices + increment).astype(np.int32)  # Performing the addition
	# to get the right indices array.
	framed_signal = padded_signal[indices.astype(np.int32, copy=False)]

	if visualisation:
		for i, frame in enumerate(framed_signal):
			plt.plot(np.linspace(i * frame_length, i * frame_length + frame_length, frame_length), frame)
		plt.title("Framed signal")
		plt.xlabel(f"Frames | Size : {win_size} s. ({frame_length} samples) ; Step : {win_step} s. ({frame_increment} samples) ; #Frames : {n_frames}")
		plt.ylabel("Frequency (Hz)")
		plt.grid()
		plt.show()
	
	return framed_signal

def dft_hamming(framed_signal: np.ndarray, nfft: int=NFFT) -> np.ndarray:
	"""
	Computes the DFT of a framed signal scaled by a hamming window.

	Parameters
	----------
	s_i: `np.ndarray`
		A framed signal of shape `(i, n)`.
	nfft: `int`
		The number of points used to perform the DFT
		(default is `512`).
	
	Returns
	-------
	dft_hamming_framed_signal: `np.ndarray`
		A 2-dimensionnal numpy array of shape `(i, nfft//2 + 1)`.
	"""

	hamming_framed_signal = framed_signal * hamming(framed_signal.shape[1])
	dft_hamming_framed_signal = np.fft.rfft(hamming_framed_signal, n=nfft)  # Returns only nfft//2 + 1 coefficients,
	# using 512, we end with 257 coefficients which is the MFCC standard.
	return dft_hamming_framed_signal

def periodogram_power_spectrum(dft_signal: np.ndarray, nfft: int=NFFT, visualisation: bool=False):  # x axis label to normalize...
	"""Computes the periodogram power spectrum of the signal.

	Parameters
	----------
	dft_signal: `np.ndarray`
		A DFT of a signal.
	nfft: `int`
		The number of points used to perform the DFT
		(default is `512`).
	visualisation: `bool`
		Graphical illustration of the operation
		(default is `False`).
	
	Returns
	-------
	periodogram_pow_spec: `np.ndarray`
		A periodogram power spectrum having the same shape as the input signal.
	"""

	assert nfft > 0, "Division by zero error.\
	The number of points required to perform the DFT must be greater than zero."
	
	periodogram_pow_spec = np.abs(dft_signal)**2 / nfft

	if visualisation:
		for i, dft_frame in enumerate(periodogram_pow_spec):
		 	plt.plot(np.linspace(i * len(dft_frame), i * len(dft_frame) + len(dft_frame), len(dft_frame)), dft_frame)
		plt.grid()
		plt.title("Periodogram")
		plt.xlabel("A.U.")
		plt.ylabel("Spectrum")
		plt.show()

	return periodogram_pow_spec

def mel_filterbank(
	fs: int,
	n_filters: int=26,
	lower_frequency: int=0,
	nfft: int=NFFT,
	visualisation: bool=False) -> np.ndarray:
	"""
	Creates a given number of Mel filterbank

	fs: `int`
		The sampling rate at which the signal is recorded.
		It is used to compute the highest mel frequency possible.
	n_filters: `int`
		Number of filters created
		(default is `26`).
	lower_frequency: `int`
		Lowest frequency in the signal, in Hertz
		(default is 0).
	nfft: `int`
		The number of points used to compute the DFT
		(default is `512`).
	visualisation: `bool`
		Graphical illustration of the operation
		(default is `False`).
	
	Returns
	-------
	filterbank: `np.ndarray`
		A 2-dimensionnal array of shape `(n_filters, nfft // 2 + 1)`
		representing the filters.
	"""

	assert lower_frequency >= 0, "The lower frequency needs to be greater than or equal or 0."

	mel_lower_freq = hertz_to_mel(lower_frequency)
	mel_upper_freq = hertz_to_mel(fs / 2)

	mel_frequencies = np.linspace(mel_lower_freq, mel_upper_freq, n_filters + 2) # n_filters + upper and lower frequencies
	bins_frequencies = mel_to_hertz(mel_frequencies)

	nearest_fft_bins = np.floor((nfft + 1) * bins_frequencies / fs).astype(np.int32)
	
	filterbank = np.zeros([n_filters, nfft // 2 + 1])
	
	for m in range(1, n_filters + 1):
		left_m = nearest_fft_bins[m - 1]
		middle_m = nearest_fft_bins[m]
		right_m = nearest_fft_bins[m + 1]

		for k in range(left_m, middle_m):
			filterbank[m - 1, k] = (k - nearest_fft_bins[m - 1]) / (nearest_fft_bins[m] - nearest_fft_bins[m - 1])
		for k in range(middle_m, right_m):
			filterbank[m - 1, k] = (nearest_fft_bins[m + 1] - k) / (nearest_fft_bins[m + 1] - nearest_fft_bins[m])
	
	if visualisation:
		for i, _filter in enumerate(filterbank):
			plt.plot(np.linspace(0, len(_filter), len(_filter)), _filter)
		plt.title(f"Mel filterbank ({n_filters} filters)")
		plt.xlabel(f"Filters length (NFFT//2 + 1, i.e. {len(_filter)} values)")
		plt.ylabel("Amplitude")
		plt.grid()
		plt.show()

	return filterbank

def spectrogram(periodogram_pow_spec: np.ndarray, mel_fbank: np.ndarray, visualisation: bool=False):
	"""
	Computes the Mel Frequency Cepstrum

	periodogram_pow_spec: `np.ndarray`
		The periodogram power spectrum used to compute the spectrogram.
	mel_fbank: `np.ndarray`
		The filterbank used to compute the spectrogram.
	visualisation: `float`
		Graphical illustration of the operation
		(default is `False`).

	Returns
	-------
	specgram: `np.ndarray`
		A 2-dimensionnal array of time-frequency	
		representation of shape `(mel_fbank.shape[0], periodogram_pow_spec.shape[0])`.
	"""

	assert mel_fbank.shape[1] == periodogram_pow_spec.shape[1], "Shape between the filterbank and\
	the periodograme power spectrum do not match. First axis of mel_fbank and periodogram_pow_spec should be equal."

	specgram = 20 * np.log10(np.dot(periodogram_pow_spec, mel_fbank.T))  # Applying the Mel filterbank to
	# the signal and converting it do dB.

	if visualisation:
		rotated_specgram = specgram.T
		plt.imshow(rotated_specgram, cmap="jet", origin="lower", aspect="auto")
		
		xlength = rotated_specgram.shape[1]
		xticks = np.arange(0, xlength, 50)  # Label every half second.
		xlabels = np.arange(0, len(xticks) / 2, 0.5)  # Converting ticks to seconds.
		if xlength > xticks[-1]:  # Adding last tick and label.
			xticks = np.append(xticks, xlength)
			xlabels = np.append(xlabels, xlength / 100)
		plt.xticks(xticks, xlabels)

		ylength = rotated_specgram.shape[0]
		yticks = np.arange(0, ylength, 5)  # Label every 5 units.
		ylabels = np.arange(0, len(yticks) / 2, 0.5)  # Converting units to kHz.
		plt.yticks(yticks, ylabels)  # Try to add last label...
		
		plt.title("Spectrogram")
		plt.xlabel("Time (s)")
		plt.ylabel("Frequency (kHz)")
		plt.colorbar()
		plt.show()

	return specgram

def cepstral_coefficients(
	specgram: np.ndarray,
	n_coeffs: int=12,
	liftering: bool=True,
	visualisation: bool=False) -> np.ndarray:
	"""
	Computes the Mel Frequency Cepstral Coefficients (MFCC).

	specgram: `np.ndarray`
		A 2-dimensionnal array representing the spectrogram of the signal.
	n_coeffs: `int`
		The number of coefficients returned.
		(default is `12`).
	liftering: `bool`
		Sinusoidal liftering. Enables better ASR in noisy signals.
		(default is `True`).
	visualisation: `bool`
		Graphical illustration of the operation
		(default is `False`).
	mfcc : `np.ndarray`
		A 2-dimensionnal array of shape `(n_frames, n_coeffs)`
		filled with the cepstral coefficients for each frame.
	"""
	
	assert n_coeffs > 0, "Number of returned coefficients must be greater than 1."
	assert len(specgram.shape) == 2, "Spectrogram input should be a 2-dimensionnal array."

	mfcc = scipy.fftpack.dct(specgram)[:, 1 : n_coeffs + 1]  # Computing DCT and only keeping \
	# the right amount of coefficients, conforming to the ASR by MFCC method.

	if liftering:  # Allows for better ASR.
		lift_coeff = 22
		n_coeff = mfcc.shape[1]

		coeff_indexes = np.arange(n_coeff)
		lift = 1 + (lift_coeff / 2) * np.sin(np.pi * coeff_indexes / lift_coeff)
		mfcc *= lift

	if visualisation:
		rotated_mfcc = mfcc.T
		plt.imshow(rotated_mfcc, cmap="jet", origin="lower", aspect="auto")

		xlength = rotated_mfcc.shape[1]
		xticks = np.arange(0, xlength, 50)  # Label every half second.
		xlabels = np.arange(0, len(xticks) / 2, 0.5)  # Converting ticks to seconds.
		if xlength > xticks[-1]:  # Adding last tick and label.
			xticks = np.append(xticks, xlength)
			xlabels = np.append(xlabels, xlength / 100)
		plt.xticks(xticks, xlabels)

		ylength = rotated_mfcc.shape[0]
		yticks = np.arange(0, ylength + 1, 2)  # Label every 2 units.
		ylabels = np.arange(0, len(yticks), 1) * 2 + 1  # Labeling filters.
		if ylabels[-1] > ylength:  # Making sure last label does stick out from \
			# the plot and shows the right value.
			yticks[-1] = yticks[-1] - 1
			ylabels[-1] = ylength
		plt.yticks(yticks, ylabels)  # Try to add last label...

		plt.title("Mel Frequency Cepstral Coefficients over time")
		plt.xlabel("Time (s)")
		plt.ylabel("MFCC")
		plt.colorbar()
		plt.show()
	
	return mfcc

def deltas(coefficients: np.ndarray, distance: int=2) -> np.ndarray:
	"""
	Computes the delta coefficients, the derivatives of MFCC in regard to frames.
	Delta-cepstral features give information about the dynamics of the MFCC coefficients over time.

	Parameters
	---------
	coefficients: `np.ndarray`
		A 2-dimensionnal array representing the coefficients for which deltas should be computed.
	distance: `int`
		The distance, in frames, taken to compute each delta
		(default is `2`).
	
	Returns
	-------
	deltas: `np.ndarray`
		A 2-dimensionnal array of shape `(coefficients.shape[0], coefficients.shape[1])`
		representing the delta-cepstral features.
	"""
	
	assert len(coefficients.shape) == 2, "The coefficient array should be 2-dimensionnal."
	assert distance > 0, "Distance parameter should be an integer greater than 0."

	# Padding the signal in order to compute the delta coefficients.
	front_padding = np.tile(coefficients[0], (distance, 1))
	back_padding = np.tile(coefficients[-1], (distance, 1))
	padded_coefficients = np.append(front_padding, coefficients, axis=0)
	padded_coefficients = np.append(padded_coefficients, back_padding, axis=0)

	deltas = np.zeros((coefficients.shape[0], coefficients.shape[1]))

	for t in range(distance, len(coefficients) + distance):
		numerator, denominator = 0, 0
		for n in range(1, distance + 1):
			numerator += n * (padded_coefficients[t + n] - padded_coefficients[t - n])
			denominator += 2 * n**2
		delta = numerator / denominator
		deltas[t - distance] = delta

	return deltas

def normalise(mfcc: np.ndarray) -> np.ndarray:
	"""
	Normalises the MFCC. Mean will become 0. Standard-deviation will become 1.

	Parameters
	----------
	mfcc: `np.ndarray`
		A 2-dimensionnal array representing the MFCC to normalise.
	
	Returns
	-------
	norm_mfcc: `np.ndarray`
		The normalised MFCC which mean is 0 and standard-deviation is 1.
	"""

	avg = np.mean(mfcc)
	sigma = np.std(mfcc)
	norm_mfcc = (mfcc - avg) / sigma

	return norm_mfcc

def mfcc(
	signal:list/np.ndarray,
	fs: int,
	alpha: float=0.97,
	win_size: float=0.025,
	win_step: float=0.01,
	nfft: int=512,
	n_filters: int=26,
	lower_frequency: int=0,
	n_coeffs: int=13,
	liftering: bool=True,
	delta: bool=True,
	delta_distance: int=2,
	delta_delta: bool=True,
	normalisation: bool=True,
	visualisation: list/bool=[False] * 7) -> np.ndarray:
	"""
	Computes the MFCC.

	Parameters
	----------
	signal: `list`/`np.ndarray`
		The signal for which MFCC must be extracted.
	fs: `int`
		The frequency at which the signal has been sampled.
	alpha: `float`
		Alpha factor, the ratio by which the signal is going te be pre-emphasised usually set to 0.95 or 0.97
		(default is `0.97`).
	win_size: `float`
		The window size, the time period which defines a single frame, in seconds
		(default is `0.0025`).
	win_step: `float`
		The window step, the frequency at which a new frame starts, in seconds
		(default is `0.01`).
	nfft: `int`
		The number of points used to perform the DFT
		(default is `512`).
	n_filters: `int`
		Number of filters created
		(default is `26`).
	lower_frequency: `int`
		Lowest frequency in the signal, in Hertz
		(default is `0`).
	n_coeffs: `int`
		The number of coefficients returned
		(default is `12`).
	liftering: `bool`
		Enables sinusoidal liftering which improves ASR
		(default is `True`).
	delta: `bool`
		Enables delta-ceptsrum analysis
		(default is `True`).
	delta_distance: `int`
		The distance, in frames, taken to compute each delta.
		The same distance is used for delta-delta analysis if enabled
		(default is `2`).
	delta_delta: `bool`
		Enables delta-delta cepstrum analysis
		(default is `True`).
	normalisation: `bool`
		Normalizes the MFCC so that its mean equals 0 and its standard-deviation equals 1
		(default is `True`).
	visualisation: `bool`
		Graphical illustration of the operation
		(default is `False`).

	Returns
	-------
	mfcc : `np.ndarray`
		A 2-dimensionnal array of shape `(n_frames, n_coeffs)` representing the MFCC.
	"""

	if type(visualisation) == bool: # If visualisation is not in a list format, converts it to it.
		if visualisation:
			visualisation = [True] * 7
		else:
			visualisation = [False] * 7
	while len(visualisation) < 7: # If the length of the list is not 7, we artificially lengthen it. Visualisation is set to false.
		visualisation.append(False)

	if visualisation[0]:
		plot_signal(signal, fs)

	pre_emphasised_signal = pre_emphasis(signal, fs, alpha=alpha, visualisation=visualisation[1])

	framed_signal = frame_signal(pre_emphasised_signal, fs, win_size=win_size, win_step=win_step, visualisation=visualisation[2])

	dft_hamming_framed_signal = dft_hamming(framed_signal, nfft=nfft)

	periodogram_pow_spectrum = periodogram_power_spectrum(dft_hamming_framed_signal, nfft=nfft, visualisation=visualisation[3])

	mel_fbank = mel_filterbank(fs, n_filters=n_filters, lower_frequency=lower_frequency, nfft=nfft, visualisation=visualisation[4])

	specgram = spectrogram(periodogram_pow_spectrum, mel_fbank, visualisation=visualisation[5])

	mfcc = cepstral_coefficients(specgram, n_coeffs=n_coeffs, liftering=liftering, visualisation=visualisation[6])

	if delta:
		delta_features = deltas(mfcc, distance=delta_distance)
		mfcc = np.append(mfcc, delta_features, axis=1)
		if delta_delta:
			delta_delta_features = deltas(delta_features, distance=delta_distance)
			mfcc = np.append(mfcc, delta_delta_features, axis=1)

	if normalisation:
		mfcc = normalise(mfcc)
	
	return mfcc.T





if __name__ == "__main__":

	print("Numpy version :", np.__version__, "| Coded in version : 1.19.2")
	print("Scipy version :", scipy.__version__, "| Coded in version : 1.5.2")

	audio = os.path.join(os.getcwd(), "audio_test/chanson_gardes_suisses.wav")

	fs, signal = read(audio)
	signal_length_secs = 2 
	signal = signal[0:int(signal_length_secs * fs)]
	
	mfcc = mfcc(signal, fs, delta=True, visualisation=True)
	print(f"MFCC successfully computed ! Matrix of shape : {mfcc.shape}")
