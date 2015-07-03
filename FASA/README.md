Copyright (c) 2014 Gökhan Yildirim
Version 1.0
===========================================================================

## GENERAL INFORMATION


This code implements the Fast, Accurate, and Size-Aware (FASA) salient 
object detection technique that is explained in the following paper:

G. Yildirim, S. Süsstrunk, "FASA: Fast, Accurate, and Size-Aware Salient 
Object Detection", ACCV, 2014

Please cite the paper if you used our source code.

This code is shared for non-commercial use only. For commercial use please 
contact the [author](gokhan.yildirim@epfl.ch)


## HOW TO COMPILE?


In order to compile our code, you need to have OpenCV library installed in 
your computer. You can find more information in the following [link](http://opencv.org/)
After installing the OpenCV library, you can compile the code as follows:
```
g++ FASA.cpp -I "OpenCV include path" -L "OpenCV library path" -lopencv_highgui -lopencv_core -lopencv_imgproc -o FASA
```
Please modify the "OpenCV include path" and "OpenCV library path" according
to your computer (without the quotes). or simply using CMake
```
$ mkdir build && cd build
$ cmake .. && make
```


## HOW TO RUN?

After compiling the code, you can run the code with following variables:
```
$ FASA —i -p /path/to/input/image/folder/ -f image_format -s /path/to/output/folder/
$ FASA —v -p /path/to/input/video/file.avi -s /path/to/output/folder/
```
image_format: Format of the images to be processed

* Example: jpg

WARNING: Please check the path naming conventions (such as using "/" or
"\"). This code is written on a Mac.

If you have any questions or bug reports, please send them to gokhan.yildirim@epfl.ch
