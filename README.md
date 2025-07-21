# Python sessile drop analysis
[![Element Android Matrix room #Sessile.Drop.Analysis:matrix.vgorcum.com](https://img.shields.io/matrix/Sessile.Drop.Analysis:matrix.vgorcum.com.svg?label=%23Sessile.Drop.Analysis:matrix.vgorcum.com&logo=matrix&server_fqdn=matrix.org)](https://matrix.to/#/#Sessile.Drop.Analysis:matrix.vgorcum.com)

Made by Mathijs van Gorcum during his PhD at the Physics of Fluids group of the University of Twente.

Python script to analyse sessile drops by measuring contact angle, drop volume and contact line speed.  
This script analyses an image sequence (in the form of an avi, a tiffstack or a folder containing the images) and finds the contact angle, drop volume and the contact line speed.  
The script assumes a black and white image of the drop on the surface, where the drop is black, and the background is white.  
The script will ask for the file (or a file, in the case of a folder containing images), a crop (to increase calculation speed, and cut off any irrelevant parts) and a baseline.
We use a subpixel edge detection, either fast, with a linear interpolation between two pixels around the edge, or slow by fitting an error function around the edge.
To find the contact line position and the contact angle the detected edge is fitted with a 3rd order polynomial fit, and the slope of the baseline is also used to calculate the contact angles.
Note that the drop volume assume cylindrical symmetry and if there is a needle present, the volume of the needle is added.

A GUI using pyqtgraph and pyqt5 is available, as well as a standalone script. The old standalone script is probably beneficial when using an IDE like spyder and you want to be able to customize the inner workings of the script.

## Screenshot

![](Screenshot.png)

## Prerequisites
If you don't use the precomiled releases you'll need:
The script requires numpy, pandas, scipy, pyqt5, opencv-python, scikit-image, imageio, shapely, pyqtgraph >=0.11.0, and python-magic.

## Running the script
To use the GUI, run QT_sessile_drop_analysis.py, while the standalone script is in Old scripts/sessile_drop_analysis.py.

## Some details
* The code is written for Python 3.8
* The edge detection uses only a horizontal subpixel correction, and when fitting the errorfunction, 40 pixels left and right of the edge are used.  
* To find the contact angle and contact point a polyfit is used, but the fit is made flipping the x and y coordinates, because polyfits don't perform well for vertical lines (ie at contact angles of 90 degrees).  
* In the non-gui script, the variable k is used to set the amount of pixels used in the polyfit, by default set at 70.  
* The thresh variable is the threshold level, for the fast edge detection the value is used explicitly while for the error function fitting the value is only used to find an approximate edge, to fit the errorfunction around.  
* The contact line speed is calculated using a linear regression of the contact line position.  
* The non-gui script calculates the speed in pixels/frame, and the volume in pixels^3, so be sure to convert it.

## Contributing
Feel free to send pull requests, critique my awful code or point out any issues.

## License
This project is licensed under the GPLv3 license - see the [LICENSE](https://github.com/mvgorcum/Sessile.drop.analysis/blob/master/LICENSE) file for details

## Extension
We modified the software so it is compatible to [Labbook](https://github.com/gipplab/Electronic-Laboratory-Notebook).
In addition we added the capability to analyze DAFI (Drop Adhesion Force Instrument) videos.