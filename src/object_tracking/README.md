# bytetrack-standalone

[ByteTrack](https://github.com/ifzhang/ByteTrack) is a simple and amazing new popular SOTA method. The source code
provided by the author are very insightful from research to deployment, like tensorrt, onnx, deepstream, ncpp.

Official is quite coupled with YoloX. 
-> Implement a simple standalone ByteTracker - less dependencies as possible.

The codes extract and made minor code (not logics) refactoring for this standalone bytetrack.

All dependencies are listed in the [requirements.txt](requirements.txt). 

### Run the Example

An example with mock videos and detectors is shown to illustrate how to use it.

How to run and get mock up output similar to MOT Challenge format:
1. `python -m pip install Cython` (this step can be ignored if you already have Cython installed)
2. `pip install -r requirements.txt`
3. `python example.py`

### Marjor Differences Made

1. Make it torch independent
2. Remove all the args and make the hyper-parameters explicit
3. Clean up some not used imports
4. Clean up the imports
5. Renamed basetrack.py to base_track.py to keep names consistent

