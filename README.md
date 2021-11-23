# BASS

## What is BASS ?

BASS stands for **Basic Audio System Solutions**. It consists of a set of utilities functions that comes in handy when doing signal processing.

BASS was designed with the objective of being a utility librairy for **Machine Learning** preprocessing.

## What can BASS do ?

### ASR

For now, the main functionnalities of BASS are mainly centered around **Automatic Speech Recognition**.

### The future of BASS

In the future, BASS might get extended to a lot of different **Natural Language Processing** tasks.

## Main methods

### Computing MFCC

The `mellib.py` file contains methods to compute a **Mel Frequency Cepstral Coefficients** transform.

Its use is very simple. The user can refer to the documentation of the function for more customization.

```python
from mellib import mfcc
mfcc = mfcc(signal)
```

### Augmenting an MFCC

The `augmentation.py` file contains a class that allows an easy way to augment MFCC.
It follows the implementation described in the specaugment article published by the Google Brain team [[1]](#1).

Its use is again, very simple. The user can refer to the documentation of the function for more customization.

```python
from augmentation import Specaugment
augmented_mfcc = Specaugment.augment(mfcc)
```

## References

<a id="1">[1]</a> Park, D. S., Chan, W., Zhang, Y., Chiu, C. C., Zoph, B., Cubuk, E. D., & Le, Q. V. (2019).
Specaugment: A simple data augmentation method for automatic speech recognition.
