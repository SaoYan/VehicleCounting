# VehicleCounting
[![DOI](https://zenodo.org/badge/96367744.svg)](https://zenodo.org/badge/latestdoi/96367744)

This is my implementation of the paper [Vehicle counting based on double virtual lines](https://link.springer.com/article/10.1007/s11760-016-1038-7)  
Both C++ and Python implementation are provided.
## How to run the code
**run C++ code**
```
git clone https://github.com/SaoYan/VehicleCounting
cd VehicleCounting
make
./DVL -vid test.mp4
```
**run Python code**
```
git clone https://github.com/SaoYan/VehicleCounting
cd VehicleCounting
python VehicleCounting.py -vid test.mp4
```
## Summary report
I‘ve written a summary report, in which I summarize the author's method, analyze the test result and propose some fundamental ideas on how to improve the author's work using deep learning method. You can get the report from [this web page](https://saoyan.github.io/projects/2017-07-07-vehicle-counting).

## Result
![](https://github.com/SaoYan/VehicleCounting/blob/master/Result_Vehicle_Counting.gif)
