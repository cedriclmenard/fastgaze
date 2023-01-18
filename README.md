# FastGaze

Article: Accepted by IEEE Transactions on Neural Systems and Rehabilitation Engineering (TNSRE-2022-00336.R2)
Available in Open Access at: https://ieeexplore.ieee.org/document/10018597

Will update with link to article when available.

Related repositories:

- BlazeFace-Pytorch [https://github.com/hollance/BlazeFace-PyTorch]: face detection model from MediaPipe ported to PyTorch and used in this repo.
- Facemesh.Pytorch [https://github.com/thepowerfuldeez/facemesh.pytorch]: face alignment model from MediaPipe ported to PyTorch.
- Irislandmarks.Pytorch [https://github.com/cedriclmenard/irislandmarks.pytorch]: iris landmarks detection model from MediaPipe ported to PyTorch and used as a basis for this work.

It should be noted that this repo, while having been cleaned-up a bit, still contains a lot of prototyping and experimenting code. A thorough clean up is in progress which should hopefully make the code clearer, but for the sake of fast dissemination, we opted to publish this repo as-is. 

There is a drastic need of documentation and code annotations, but this will come in due time.

**NOTE ON ISSUES**

If you encounter any issues while trying to setup and use this work, please let us know by writing an issue here in Github. We will do our best to respond to it quickly and either suggest a fix or implement a fix.

## 0. Setup and requirements

Only tested on Python 3.9.

Simply install the required dependencies:

```bash
python3 -m pip install -r requirements.txt
```

## 1. Generating the datasets

To generate the dataset, make sure the dataset data is available in a local directory (extracted from the original compressed file if it's the case) and use the `generate_hdf5.py` script. Replace the `dataset_path` argument with path to the folder containing the dataset.

Example for mpiigaze:

```bash

python3 generate_hdf5.py dataset_path mpiigaze ~/datasets/hdf5/mpiig.hdf5 --annotation_file --undistort

```

Example for utmultiview:

```bash

python3 generate_hdf5.py dataset_path utmultiviewfromsynth ~/datasets/hdf5/utmvfromsynth.hdf5 --annotation_file --undistort

```

Example for gazecapture:

```bash

python3 generate_hdf5.py dataset_path gazecapturefaze ~/datasets/hdf5/gc.hdf5 --annotation_file --undistort

```

The `--annotation_file` option without arguments makes use of the normalization technique (de-rolling) as generally used in other gaze estimation papers. See the original article for more details.

## 2. Training

Training can be done by using the helper scripts:

```bash

./scripts/run_mpiig_leave_one_out.sh
./scripts/run_utmv_leave_one_out.sh
./scripts/run_gc_faze.sh
```


## 3. Validation
Once training is done, you can run the following bash commands to print out the overall results:

MPIIGaze
```bash
for i in {0..14}; do tail -n 3 mpiig_run_idx_$i.out | grep -oP "(?<={\'test_error\': )[+-]?([0-9]*[.])?[0-9]+(?=, )"; done
```

UTMultiview
```bash
for i in {0..2}; do tail -n 3 utmv_run_3fold_idx_$i.out | grep -oP "(?<={\'test_error\': )[+-]?([0-9]*[.])?[0-9]+(?=, )"; done
```

For GazeCapture, simply look at the output in `gc_run.txt` from the training, as there are now k-fold or leave one out validation scheme.

## 4. Realtime Demo

Please note that the default trained model is trained for an application that is not gaze estimation of a point on a computer screen. As such, their might be some issues currently with the realtime demo results. I will update the model with a properly trained model when I have time.

See the `--help` option for the following help when running the realtime demo:

```
usage: realtime_demo.py [-h] [-c [CALIBRATION]] [--cpu] [-m MODEL] camera_idx [fovx]

Real-time gaze estimation demo

positional arguments:
  camera_idx            OS index of camera to use.
  fovx                  Horizontal FOV in degrees of camera. Does not matter if calib file is given.

optional arguments:
  -h, --help            show this help message and exit
  -c [CALIBRATION], --calibration [CALIBRATION]
                        Use camera calibration.
  --cpu                 Only use the CPU (disables GPU acceleration).
  -m MODEL, --model MODEL
                        Specify a model path (pth). Defaults to default path specified in source.
```

Note that the camera calibration option `-c` requires a YAML calibration file as generally obtained through OpenCV. Otherwise, by specifying the horizontal FoV of the camera through the `fovx` argument, a basic calibration matrix is used. This will be far less accurate, but can be a good approximation for debugging or demonstration purposes.

The `-m` option allows you to specify the path of a trained model instead of relying on the default one.

