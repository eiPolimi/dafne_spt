# Simulation Processing Toolbox for the Dafne Project
A collection of Python scripts and libraries to post-process the simulation output of the Dafne Project in order to perform indicator computation and to export data to Dafne Geoportal and Multi Perspective Visual Analytics Tool. 

## Content
The Toolbox is written in Python language and composed by a set of script and libraries:
* *globalparameter.py* declares a dictionary of input/output path and of useful parameters, imported by the other script of the toolbox; 
* *dafneutils.py* translates the system model into Python classes of components, variables and indicators, ready to be populated. It also defines a number of utility functions, imported by the other script of the toolbox;
* *sim_importer.py* imports the outcomes of DAF/WEF simulations into a python object (.pkl) including all the components time series and saving it into the file system. Different output formats of DAF/WEF simulations are supported by setting up customized dictionary within the script;
* *indicators_lib.py* declares a library of functions for indicators computation;
* *calc_indicators.py* takes as input the system model created by the sim_importer.py and computes indicators at different time and spatial scales. It generates as output a number of time series (.csv) and images (.png) files, for both variables and indicators. Time series output are in the DAFNE Geoportal format, which combines data and metadata. This script creates also output to source the Multi-perspective Visual Analytics tool, developed by EIPCM1;
* *load_procedures.py* takes as input the time series in the csv format produced by the calc_indicators.py and it uploads them in the DG database.

## Credits
This toolbox has been developed by Marco Micotti, Environmental Intelligence group,  Politecnico di Milano, with contribution by Simone Corti.

The DAFNE H2020 Project (Decision Analytic Framework to explore the water-energy-food NExus in complex transboundary water resources of fast developing countries), has been funded by the European Commission. 
