# Audio-based-activity-recognization
A machine learning pipeline to detect activities and events using sound.

The project involves data collection, data pre-processing/signal conditioning, feature extraction, using an existing ML implementation, and analysis of results for audio-based activity recognization.

## Data collection
The data used in the project are collected with Voice Recorder on an iPhone. 20 samples of five events (approximately 30 seconds each) including microwave, blender, fire alarm, vacuum cleaner, and music are collected to build the model from scratch. 20 samples of silence are also included. They are used to develop a logic that can be used in the future to filter out silent periods or segment actual events.

The data collected can be found [here]().

## Feature Engineering

### Pre-processing
- Trim and make each file the same length (30 seconds each)
- Normalize the amplitude of audio files so the value stays between 0 and 1 -- so we won't reply on loudness as a feature since it changes with distance
- Convert the raw time-domain signal into frequency-domain and normalize the amplitude using `librosa.stft`
### Feature extraction
- **Domain-specific features**
  - *Standard deviation of time-domain signal*: how loudness changes over time in a recording
  - *Standard deviation of frequency-domain signal*: how diverse the frequency spectrum is. This feature should be extremely useful for fire alarms and music since their frequencies vary a lot over the recording.
  - *Median of frequency-domain signal*: the median frequency should represent the main frequency level of white-noise classes (blender, vacuum, silence, and main microwave sound). For the rest, it should represent the overall frequency of the data to help us differentiate.
  - *Average zero-crossing rate of time-domain signal*: the rate at which a signal changes from positive to zero to negative or from negative to zero to positive. This feature aims the capture the change in voltage and should work well with the music.
  - Note: for frequency-domain signal, since we want to observe the overall trend, the calculation is based on the weighted average frequency of each frame given by `librosa.feature.spectral_centroid`, where each frame of a magnitude spectrogram is normalized and treated as a distribution over frequency bins, from which the mean (centroid) is extracted per frame.
- **Spectrogram features**
  - *Mel-frequency cepstral coefficients (MFCCs)*: we bin the spectrogram data from the recordings and use each bin as a feature by using mfcc (convert the 2D array of samples given by stft in pre-processing into a smaller array)
  - By calculating MFCC, we window the signal, apply the DFT, take the log of the magnitude, and then warp the frequencies on a Mel scale, followed by applying the inverse DCT. Basically, we have extracted features of each bin on a Mel scale.
  

## Model Training
The model is trained using Random Forest Classifer from `sklearn`.
***Observations***: The model seems to be overfit. This might happen because for most of the data, we repeatly recorded the same infomation so there's not that much variety.
