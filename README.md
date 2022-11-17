# EU_HFR_NODE_pyHFR
Python3 toolbox for the operational workflow of the European HFR Node (EU HFR Node). Tools for the centralized processing at the EU HFR Node and for local processing on provider side.

This toolbox is based on the HFRadarPy toolbox from ROWG (https://github.com/rowg/HFRadarPy), that was refined and integrated to manage both files written in CODAR Tabular Format (CTF) and files wirtten in the WERA and LERA crad_ascii and cur_Asc native formats, and to perform wighted least square combination of radial currents into total currents as defined in "Gurgel, K-W., Shipborne measurement of surface current fields by HF radar. Proceedings of OCEANS'94. Vol. 3. IEEE, 1994".

These applications are written in Python3 language and the architecture of the workflow is based on a MySQL database containing information about data and metadata. The applications are designed for High Frequency Radar (HFR) data management according to the European HFR node processing workflow, thus generating radial and total velocity files in netCDF format according to the European standard data and metadata model for near real time HFR current data.

The database is composed by the following tables:
- account_tb: it contains the general information about HFR providers and the HFR networks they manage.
- network_tb: it contains the general information about the HFR network producing the radial and total files. These information will be used for the metadata content of the netCDF files.
- station_tb: it contains the general information about the radar sites belonging to each HFR network producing the radial files. These information will be used for the metadata content of the netCDF files.
- radial_input_tb: it contains information about the radial files to be converted and combined into total files.
- radial_HFRnetCDF_tb: it contains information about the converted radial files.
- total_input_tb: it contains information about the total files to be converted.
- total_HFRnetCDF_tb: it contains information about the combined and converted total files.

The applications are intended to:
- load radial files information onto the database in table radial_input_tb;
- load total files information onto the database in table total_input_tb;
- convert Codar native .tuv files and WERA or LERA native .cur_asc files for total currents into the European standard data and metadata model for near real time HFR current data;
- convert Codar native .ruv files and WERA or LERA native .crad_ascii files for radial currents into the European standard data and metadata model for near real time HFR current data and combine them for generating total current files according to the European standard data and metadata model for near real time HFR current data.

General information for the tables network_tb and station_tb are loaded onto the database via a webform to be filled by the data providers. The webform is available at https://webform.hfrnode.eu

All generated radial and total netCDF files are quality controlled according the the QC tests defined as standard for the European HFR node and for the data distribution on CMEMS-INSTAC and SeaDataNet platforms.

The whole workflow is intended to run automatically to continuously convert and combine near real time HFR data produced by data providers. 

The application EU_HFR_NODE_NRTprocessor.py is intended for centralized processing at the EU HFR Node and operates the functions for collecting and processing HFR radial and total data for all the networks connected to the node.

The application local_NRTprocessor.py is intended for local processing on provider's side and operates the functions for collecting and processing HFR radial and total data for the network of the specific provider. The providers using this application have the responsibility of managing the operational synchronization of the generated netCDF in the the European standard data and metadata model towards the EU HFR Node.

Usage: EU_HFR_NODE_NRTprocessor.py -m [number of days in the past when to start processing (default to 3)]

The required packages are:
- pandas
- sqlalchemy
- mysql-connector-python
- xarray
- glob
- pyproj.Geod

The guidelines on how to synchronize the providers' HFR radial and total data towards the EU HFR Node are available at ​https://doi.org/10.25704/9XPF-76G7
How to cite:
- when using these guidelines, ​please use the following citation carefully and correctly​:
Reyes, E., Rotllán, P., Rubio, A., Corgnati, L., Mader, J., & Mantovani, C. (2019).
Guidelines on how to sync your High Frequency (HF) radar data with the European HF
Radar node (Version 1.1). Balearic Islands Coastal Observing and Forecasting System,
SOCIB . https://doi.org/10.25704/9XPF-76G7

Cite as:
Lorenzo Corgnati. (2022). EU_HFR_NODE_pyHFR. DOI to be assigned.


Author: Lorenzo Corgnati

Date: November 15, 2022

E-mail: lorenzo.corgnati@sp.ismar.cnr.it
