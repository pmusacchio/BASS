import os
import numpy as np
import matplotlib.pyplot as plt
import mellib
import soundfile as sf





class Specaugment:
	"""
	Contains MFCC augmentation tools.

	Methods
	-------
	augment:
		Augments the given MFCC for the desired domains.

	Usage
	-----
	>>> augmented_mfcc = Specaugment.augment(mfcc)
	"""

	TIME_MASKING: bool = True
	MEL_MASKING: bool = True
	MAX_TIME_MASK: int = 2
	MAX_MEL_MASKING: int = 2
	TIME_MASK: int = 80
	MEL_MASK: int = 2

	def __init__(self):
		# Attributes are only set and used by the differents methods.
		self.avg = None
		self.n_freq = None
		self.n_time = None

	@classmethod
	def __time_masking(
		self,
		mfcc: np.ndarray,
		max_mask: int=MAX_TIME_MASK,
		time_mask: int=TIME_MASK) -> np.ndarray:
		"""
		Performs a time augmentation on a given MFCC.

		Parameters
		----------
		mfcc: `np.ndarray`
			A 2-dimensionnal array representing the MFCC over time of the signal.
		max_mask: `int`
			The maximum number of slices allowed for augmenting the signal in the time domain
			(default is `3`).
		time_mask: `int`
			The maximum size allowed for a slice in the time domain
			(default is `20`).
		
		Returns
		-------
		mfcc: `np.ndarray`
			The time-augmented MFCC.
		"""

		nb_mask = np.random.randint(0, max_mask + 1)

		for _ in range(nb_mask):
			time_mask_size = np.random.randint(0, time_mask + 1)
			lower_time_mask = np.random.randint(0, self.n_time - time_mask + 1)
			upper_time_mask = lower_time_mask + time_mask_size
			mfcc[:, lower_time_mask : upper_time_mask + 1] = np.full(mfcc[:, lower_time_mask : upper_time_mask + 1].shape, self.avg)

		return mfcc

	@classmethod
	def __mel_masking(
		self,
		mfcc: np.ndarray,
		max_mask: int=MAX_MEL_MASKING,
		mel_mask: int=MEL_MASK) -> np.ndarray:
		"""
		Performs a frequency augmentation on a given MFCC.

		Parameters
		----------
		mfcc: `np.ndarray`
			A 2-dimensionnal array representing the MFCC over time of the signal.
		max_mask: `int`
			The maximum number of slices allowed for augmenting the signal in the frequency domain
			(default is `3`).
		mel_mask: `int`
			The maximum size allowed for a slice in the frequency domain
			(default is `3`).

		Returns
		-------
		mfcc: `np.ndarray`
			The frequency-augmented MFCC.
		"""
		
		nb_mask = np.random.randint(0, max_mask + 1)

		for _ in range(nb_mask):
			frequency_mask_size = np.random.randint(0, mel_mask + 1)
			lower_frequency_mask = np.random.randint(0, self.n_freq - mel_mask + 1)
			upper_frequency_mask = lower_frequency_mask + frequency_mask_size
			mfcc[lower_frequency_mask : upper_frequency_mask + 1] = np.full(mfcc[lower_frequency_mask : upper_frequency_mask + 1].shape, self.avg)

		return mfcc

	@classmethod
	def augment(
		self,
		mfcc: np.ndarray,
		time_masking: bool=TIME_MASKING,
		mel_masking: bool=MEL_MASKING,
		max_time_mask: int=MAX_TIME_MASK,
		max_mel_masking: int=MAX_MEL_MASKING,
		time_mask: int=TIME_MASK,
		mel_mask: int=MEL_MASK,
		visualisation: bool=False) -> np.ndarray:
		"""
		Augments the given MFCC for the desired domains.

		Parameters
		----------
		mfcc: `np.ndarray`
			A 2-dimensionnal array representing the MFCC over time of the signal. 
		time_masking: `bool`
			Enables time masking augmentation
			(default is `True`).
		mel_masking: `bool`
			Enables frequency masking augmentation
			(default is `True`).
		max_time_mask: `int`
			The maximum number of time slices that can be augmented
			(default is `2`).
		max_mel_masking: `int`
			The maximum number of frequency slices that can be augmented
			(defualt is `2`).
		time_mask: `int`
			The maximum size of an augmented slice in the time domain
			(default is `30`).
		:param mel_mask: `int`
			The maximum size of an augmented slice in the frequency domain
			(default is `2`).
		visualisation: `bool`
			Graphical illustration of the operation
			(default is `False`).

		Returns
		-------
		mfcc: `np.ndarray`
			The augmented MFCC.
		"""

		if time_masking:
			mfcc = Specaugment.__time_masking(mfcc, max_mask=max_time_mask, time_mask=time_mask)
		if mel_masking:
			mfcc = Specaugment.__mel_masking(mfcc, max_mask=max_mel_masking, mel_mask=mel_mask)


		if visualisation:
			plt.imshow(mfcc, cmap="jet", origin="lower", aspect="auto")
			plt.title("Augmented MFCC over time")
			plt.xlabel("Time (s)")
			plt.ylabel("MFCC")
			plt.colorbar()
			plt.show()

		return mfcc





if __name__ == "__main__":

	path = os.path.join(os.getcwd(), "audio_test/chanson_gardes_suisses.wav")
	sig, sr = sf.read(path)
	sig = sig[:int(sr * 2.5)]
	mfcc = mellib.mfcc(sig, sr, visualisation=False)
	plt.imshow(mfcc, aspect="auto", origin="lower", cmap="jet")
	plt.show()
	specaug = Specaugment.augment(mfcc, visualisation=True)