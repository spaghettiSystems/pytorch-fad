# Fréchet Audio Distance (FAD) PyTorch Implementation

This is a port of the official implementation of [Fréchet Audio Distance](https://arxiv.org/abs/1812.08466) to PyTorch. 
See [here](https://github.com/google-research/google-research/tree/master/frechet_audio_distance) for the original implementation using Tensorflow v1 and Beam.

To use, simply call fad_score.py and pass in 2 paths to folders containing the files you wish to compare. Set the recursive flag as appropriate.

```
fid_score.py path/to/dataset path/to/generated --recursive False
```

Alternatively, instantiate FADMetric from fad.py.

device: specifies what device to run the computations on, 'cuda' or 'cpu'.

base_path: specifies the test set to compare against. recommended to set this to avoid repeated computations.

recursive: specifies whether base_path is to be checked recursively for .wav files

Example usage:
```
metric = FADMetric(device='cpu', base_path = "dataset/test/path")
fad_score = metric.compare_base_to_path("generated/samples/path")
print(fad_score)
```

# Warning
This implementation seems to produce slightly different results compared to the original.

On the [test files from the original repo](https://github.com/google-research/google-research/blob/master/frechet_audio_distance/gen_test_files.py) the difference in score is as follows:
```
Pytorch FAD:  4.724766023409964
Original TF FAD: 4.642469

PyTorch FAD:  14.0606470686105
Original TF FAD: 12.742362
```

The cause is still under investigation. Pull requests welcome. :P

# Known Issues
- The repository loads and processes audio files sequentially without batching. This should be trivial to improve, hopefully in a future version.
- ~~Setting the device to use for the computation (cpu/gpu) is probably broken for now~~
- ~~Not pip-installable~~

# Credits
We use the VGGish port here: https://github.com/harritaylor/torchvggish

Repository is based very heavily on: https://github.com/mseitzer/pytorch-fid

WAV dataset class based on the one in this lovely repo: https://github.com/archinetai/audio-diffusion-pytorch