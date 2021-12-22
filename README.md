# Recalibration of SL2P

This repository contains work done on the Recalibration of SL2P using Neural Networks that was done over the course of the Fall 2021 term (September 2021-December 2021).

The main objective of the project was to reduce the bias towards low values (<=2) of Leaf Area Index by the SL2P algorithm. The way this was done was by modifying the prior distributions for each of the different variables we are interested in studying (e.g. changing the distibution from a normal distribution to a uniform one) and then using the new  calibration data to train a neural network.  

The variables that were anlyzed were Leaf Area Index (LAI), Fraction of Cover (FCOVER), and  Fraction of Absorbed Photosynthetically Active Radiation (FAPAR). More information about these variables can be found at:
* https://step.esa.int/docs/extra/ATBD_S2ToolBox_L2B_V1.1.pdf 

Google Earth Engine was a tool used towards the end of the project, more information about Google Earth Engine can be found in the following document:
* https://www.sciencedirect.com/science/article/pii/S0034425717302900 

## Environment Setup

To install Anaconda, use the following guide:
https://www.anaconda.com/products/individual

Create the following environment using the following command: 
 <br />
`conda create -n eeALR ipython jupyterlab numpy scipy pandas matplotlib scikit-learn tensorflow`

The following packages should be installed as follows:

`conda install -c conda-forge earthengine-api -y`

`conda install -c conda-forge folium -y`

`conda install geemap -c conda-forge`

Activate the new environment:

`conda activate eeALR`

Verify that the new environment was installed correctly:

`conda env list`

There should be an asterisk next to eeALR showing that it is the active environment.

You can now launch Jupyter Lab (a newer web based IDE for jupyter notebboks) in the current directory with the cloned git repository.

`jupyter lab`

Make sure you are always in the correct environment (with the necessary packages) before launching jupyter lab.

## Authentication of Earth Engine

When running Google Earth Engine on a new machine, you should run the authentication flow as follows:
 <br />
`import ee` 
 <br />
`ee.Authenticate()`
 <br />
`ee.Initialize()`

This only needs to be done once, afterwards, only the following lines are needed to use the Google Earth Engine API:
 <br />
`import ee` 
 <br />
`ee.Initialize()`

## Important Notes

* `ALR_functions.py`, `ee_functions.py`, `feature_collections.py`, `image_bands.py`, and `wrapper_nets.py` are needed so that `apply_nnet_image.ipynb` can function properly 
* `recalibrate_nnet.ipynb` and `apply_nnet_image.ipynb` are the most important notebooks since they essentially represent the culmination of the work done in this project
* Meanwhile, the rest of the notebooks represent the workflow over the course of the term. 
