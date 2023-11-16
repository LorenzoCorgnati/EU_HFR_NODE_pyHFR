#!/usr/bin/python3


# Created on Wed Nov 15 17:22:46 2023

# @author: Lorenzo Corgnati
# e-mail: lorenzo.corgnati@sp.ismar.cnr.it


# This application reads from the EU HFR NODE EU HFR NODE database the information about 
# radial and total HFR files (both Codar and WERA) pushed by the data providers,
# combines radials into totals, generates HFR radial and total data to netCDF 
# files according to the European standard data model for data distribution towards
# EMODnet Physics portal and  generates HFR radial and total data to netCDF 
# files according to the Copernicus Marine Service data model for data distribution towards
# Copernicus Marine Service In Situ component.

# This application works on historical data, i.e. it processes HFR data within time intervals
# specified by the user. It does not work for Near Real Time operations.

# When calling the application it is possible to specify if all the networks have to be processed
# or only the selected one, the time interval to be processed and if the generation of HFR radial 
# and total netCDF data files according to the Copernicus Marine Service data model has to
# be performed.

# This application implements parallel computing by launching a separate process 
# per each HFR network to be processed (in case of processing multiple networks).

import os
import sys
import getopt
import glob
import logging
import datetime as dt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import sqlalchemy
from sqlalchemy import text
from dateutil.relativedelta import relativedelta
from radials import Radial, buildEHNradialFolder, buildEHNradialFilename, convertEHNtoINSTACradialDatamodel, buildINSTACradialFolder, buildINSTACradialFilename
from totals import Total, buildEHNtotalFolder, buildEHNtotalFilename, combineRadials, convertEHNtoINSTACtotalDatamodel, buildINSTACtotalFolder, buildINSTACtotalFilename, buildUStotal
from calc import createLonLatGridFromBB, createLonLatGridFromBBwera, createLonLatGridFromTopLeftPointWera
from common import addBoundingBoxMetadata
import pickle
from multiprocessing import Process
import time
import xarray as xr
import netCDF4 as nc4

######################
# PROCESSING FUNCTIONS
######################

def modifyNetworkDataFolders(ntwDF,dataFolder,logger):
    """
    This function replaces the data folder paths in the total_input_folder_path and in the 
    total_HFRnetCDF_folder_path fields of the DataFrame containing information about the network
    according to the data folder path specified by the user.
    
    INPUT:
        ntwDF: Series containing the information of the network
        dataFolder: full path of the folder containing network data
        logger: logger object of the current processing
        
    OUTPUT:
        ntwDF: Series containing the information of the network
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    mfErr = False
    
    try:
        # Check if the total_input_folder_path field is specified
        if ntwDF.loc['total_input_folder_path']:
            # Modify the total_input_folder_path
            ntwDF.loc['total_input_folder_path'] = os.path.join(dataFolder,ntwDF.loc['network_id'],ntwDF.loc['total_input_folder_path'].split('/')[-1])
            
        # Check if the total_HFRnetCDF_folder_path field is specified
        if ntwDF.loc['total_HFRnetCDF_folder_path']:
            # Modify the total_HFRnetCDF_folder_path
            ntwDF.loc['total_HFRnetCDF_folder_path'] = os.path.join(dataFolder,ntwDF.loc['network_id'],'Totals_nc')
        
    except Exception as err:
        mfErr = True
        logger.error(err.args[0] + ' in modifying total folder paths for network ' + ntwDF.loc['network_id'])
    
    return ntwDF

def modifyStationDataFolders(staDF,dataFolder,logger):
    """
    This function replaces the data folder paths in the radial_input_folder_path and in the 
    radial_HFRnetCDF_folder_path fields of the DataFrame containing information about the radial
    stations according to the data folder path specified by the user.
    
    INPUT:
        staDF: Series containing the information of the radial station
        dataFolder: full path of the folder containing network data
        logger: logger object of the current processing
        
    OUTPUT:
        staDF: Series containing the information of the radial station
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    mfErr = False
    
    try:
        # Check if the radial_input_folder_path field is specified
        if staDF.loc['radial_input_folder_path']:
            # Modify the radial_input_folder_path
            staDF.loc['radial_input_folder_path'] = os.path.join(dataFolder,staDF.loc['network_id'],staDF.loc['radial_input_folder_path'].split('/')[-2],staDF.loc['radial_input_folder_path'].split('/')[-1])
            
        # Check if the radial_HFRnetCDF_folder_path field is specified
        if staDF.loc['radial_HFRnetCDF_folder_path']:
            # Modify the radial_HFRnetCDF_folder_path
            staDF.loc['radial_HFRnetCDF_folder_path'] = os.path.join(dataFolder,staDF.loc['network_id'],'Radials_nc')
        
    except Exception as err:
        mfErr = True
        logger.error(err.args[0] + ' in modifying radial folder paths for ' + staDF.loc['network_id'] + '-' + staDF.loc['station_id'] + ' station')
    
    return staDF

def createTotalFromUStds(ts,pts,USxds,networkData,stationData,vers,logger):
    """
    This function creates a Total object for each timestamp from the input aggregated xarray dataset read
    from the US TDS. The Total object is saved as .ttl file.
    
    INPUT:
        ts: timestamp as timestamp object
        pts: Geoseries containing the lon/lat positions of the data geographical grid
        USxds: xarray DataSet containing gridded total data related to the input timestamp
        networkData: DataFrame containing the information of the network to which the total belongs
        stationData: DataFrame containing the information of the radial sites that produced the total
        vers: version of the data model
        logger: logger object of the current processing
        
    OUTPUT:
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    utErr = False
    
    try:
    
    #####
    # Build the Total object
    #####  

        # Convert timestamp to datetime
        ts = ts.to_pydatetime()     
    
        # Create the Total object
        Tus = buildUStotal(ts,pts,USxds,networkData,stationData)
        
    #####
    # Save Total object as .ttl file with pickle
    #####
    
        # Set the filename (with full path) for the netCDF file
        ncFilePath = buildEHNtotalFolder(networkData.iloc[0]['total_HFRnetCDF_folder_path'],ts,vers)
        ncFilename = buildEHNtotalFilename(networkData.iloc[0]['network_id'],ts,'.nc')
        ncFile = ncFilePath + ncFilename 
        
        # Add filename and filepath to the Total object
        Tus.file_path = ncFilePath.replace('nc','ttl')
        Tus.file_name = ncFilename.replace('nc','ttl')
        Tus.full_file = ncFile.replace('nc','ttl')
    
        # Create the destination folder
        if not os.path.isdir(ncFilePath.replace('nc','ttl')):
            os.makedirs(ncFilePath.replace('nc','ttl'))
            
        # Save the ttl file
        with open(ncFile.replace('nc','ttl'), 'wb') as ttlFile:
              pickle.dump(Tus, ttlFile) 
        logger.info(ncFilename.replace('nc','ttl') + ' total ttl file succesfully created and stored (' + vers + ').')
        
    except Exception as err:
        utErr = True
        logger.error(err.args[0] + ' for total file ' + ncFilename.replace('nc','ttl'))
    
    return

def applyINSTACtotalDataModel(dmTot,networkData,stationData,instacBuffer,vers,logger):
    """
    This function aggregates all the total netCDF files related to data measured in the day of the 
    Total object timestamp, applies the Copernicus Marine Service In Situ TAC standard data model 
    to the aggregated dataset and saves the resulting aggregated netCDF file into the buffer for 
    pushing data towards the Copernicus Marine Service In Situ TAC.
    
    INPUTS:
        dmTot: Series containing the total to be processed with the related information
        networkData: DataFrame containing the information of the network producing the total
        stationData: DataFrame containing the information of the radial sites belonging to the network
        instacBuffer: full path of the folder where to save data for Copernicus Marine Service 
                    (if None, no files for Copernicus Marine Service are produced)
        vers: version of the data model
        logger: logger object of the current processing

        
    OUTPUTS:
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    dmErr = False 
    
    # Get the Total object
    T = dmTot['Total']
    
    # Check if Total object contains data
    if T.data.empty:
        logger.info(T.file_name + ' total file is empty: INSTAC data model not applied')
        return
    
    # Set the filename (with full path) for the hourly netCDF file
    ncFilePath = buildEHNtotalFolder(networkData.iloc[0]['total_HFRnetCDF_folder_path'],T.time,vers)
    ncFilename = buildEHNtotalFilename(networkData.iloc[0]['network_id'],T.time,'.nc')
    
    try:        
        
    #####
    # Open the netCDF files of the day in an aggregated dataset
    #####
    
        # List all netCDF files in the current day folder
        hourlyFiles = [file for file in glob.glob(os.path.join(ncFilePath,'**/*.nc'), recursive = True)]

        if len(hourlyFiles)>0:
            # Open all netCDF files in the current day folder
            # dailyDS = xr.open_mfdataset(hourlyFiles,combine='nested',concat_dim='TIME')
            dailyDS = xr.open_mfdataset(hourlyFiles,combine='by_coords',compat='broadcast_equals')
            
    #####        
    # Convert to Copernicus Marine Service In Situ TAC data format (daily aggregated netCDF)  
    #####
        
            # Apply the Copernicus Marine Service In Situ TAC data model
            instacDS = convertEHNtoINSTACtotalDatamodel(dailyDS, networkData, stationData, vers)
            
            # # Enable compression
            # enc = {}
            # for vv in instacDS.data_vars:
            #     if instacDS[vv].ndim < 2:
            #         continue
            
            #     enc[vv] = instacDS[vv].encoding
            #     enc[vv]['zlib'] = True
            #     enc[vv]['complevel'] = 9
            #     enc[vv]['fletcher32'] = True
            
            # Set the filename (with full path) for the aggregated netCDF file
            ncFilePathInstac = buildINSTACtotalFolder(instacBuffer,networkData.iloc[0]['network_id'],vers)
            ncFilenameInstac = buildINSTACtotalFilename(networkData.iloc[0]['network_id'],T.time,'.nc')
            ncFileInstac = ncFilePathInstac + ncFilenameInstac 
            
            # Create the destination folder
            if not os.path.isdir(ncFilePathInstac):
                os.makedirs(ncFilePathInstac)
                
            # Check if the netCDF file exists and remove it
            if os.path.isfile(ncFileInstac):
                os.remove(ncFileInstac)
            
            # Create netCDF wih compression from DataSet and save it
            # instacDS.to_netcdf(ncFileInstac, format='NETCDF4_CLASSIC', engine='netcdf4', encoding=enc)    # IF COMPRESION ENABLED
            instacDS.to_netcdf(ncFileInstac, format='NETCDF4_CLASSIC', engine='netcdf4')                    # IF COMPRESSION NOT ENABLED
            
            # Modify the units attribute of TIME variable for including timezone digit
            ncf = nc4.Dataset(ncFileInstac,'r+',format='NETCDF4_CLASSIC')
            ncf.variables['TIME'].units = 'days since 1950-01-01T00:00:00Z'
            ncf.variables['TIME'].calendar = 'standard'
            ncf.close()
        
            logger.info(ncFilenameInstac + ' total netCDF file succesfully created and stored in Copericus Marine Service In Situ TAC buffer (' + vers + ').')
            
        else:
            return

    except Exception as err:
        dmErr = True
        if 'ncFilenameInstac' in locals():
            logger.error(err.args[0] + ' in creating Copernicus Marine Service In Situ TAC total file ' + ncFilenameInstac)
        else:
            logger.error(err.args[0] + ' in creating Copernicus Marine Service In Situ TAC total file for timestamp ' + T.time.strftime('%Y-%m-%d %H:%M:%S'))
        return     
    
    return

def applyINSTACradialDataModel(dmRad,networkData,radSiteData,instacBuffer,vers,logger):
    """
    This function aggregates all the radial netCDF files related to data measured in the day of the 
    Radial object timestamp, applies the Copernicus Marine Service In Situ TAC standard data model 
    to the aggregated dataset and saves the resulting aggregated netCDF file into the buffer for 
    pushing data towards the Copernicus Marine Service In Situ TAC.
    
    INPUTS:
        dmRad: Series containing the radial to be processed with the related information
        networkData: DataFrame containing the information of the network to which the radial site belongs
        radSiteData: DataFrame containing the information of the radial site that produced the radial
        instacBuffer: full path of the folder where to save data for Copernicus Marine Service 
                    (if None, no files for Copernicus Marine Service are produced)
        vers: version of the data model
        logger: logger object of the current processing

        
    OUTPUTS:
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    dmErr = False
    
    # Get the Radial object
    R = dmRad['Radial']
    
    # Check if Radial object contains data
    if R.data.empty:
        logger.info(R.file_name + ' radial file is empty: INSTAC data model not applied')
        return
    
    # Set the filename (with full path) for the hourly netCDF file
    ncFilePath = buildEHNradialFolder(radSiteData.iloc[0]['radial_HFRnetCDF_folder_path'],radSiteData.iloc[0]['station_id'],R.time,vers)
    ncFilename = buildEHNradialFilename(radSiteData.iloc[0]['network_id'],radSiteData.iloc[0]['station_id'],R.time,'.nc')
    
    try:        
        
    #####
    # Open the netCDF files of the day in an aggregated dataset
    #####
    
        # List all netCDF files in the current day folder
        hourlyFiles = [file for file in glob.glob(os.path.join(ncFilePath,'**/*.nc'), recursive = True)]

        if len(hourlyFiles)>0:
            # Open all netCDF files in the current day folder
            # dailyDS = xr.open_mfdataset(hourlyFiles,combine='nested',concat_dim='TIME')
            dailyDS = xr.open_mfdataset(hourlyFiles,combine='by_coords',compat='broadcast_equals')
                
    #####        
    # Convert to Copernicus Marine Service In Situ TAC data format (daily aggregated netCDF)  
    #####
        
            # Apply the Copernicus Marine Service In Situ TAC data model
            instacDS = convertEHNtoINSTACradialDatamodel(dailyDS, networkData, radSiteData, vers)
            
            # # Enable compression
            # enc = {}
            # for vv in instacDS.data_vars:
            #     if instacDS[vv].ndim < 2:
            #         continue
            
            #     enc[vv] = instacDS[vv].encoding
            #     enc[vv]['zlib'] = True
            #     enc[vv]['complevel'] = 9
            #     enc[vv]['fletcher32'] = True
            
            # Set the filename (with full path) for the aggregated netCDF file
            ncFilePathInstac = buildINSTACradialFolder(instacBuffer,radSiteData.iloc[0]['network_id'],radSiteData.iloc[0]['station_id'],vers)
            ncFilenameInstac = buildINSTACradialFilename(radSiteData.iloc[0]['network_id'],radSiteData.iloc[0]['station_id'],R.time,'.nc')
            ncFileInstac = ncFilePathInstac + ncFilenameInstac 
            
            # Create the destination folder
            if not os.path.isdir(ncFilePathInstac):
                os.makedirs(ncFilePathInstac)
                
            # Check if the netCDF file exists and remove it
            if os.path.isfile(ncFileInstac):
                os.remove(ncFileInstac)
            
            # Create netCDF wih compression from DataSet and save it
            # instacDS.to_netcdf(ncFileInstac, format='NETCDF4_CLASSIC', engine='netcdf4', encoding=enc)    # IF COMPRESSION ENABLED
            instacDS.to_netcdf(ncFileInstac, format='NETCDF4_CLASSIC', engine='netcdf4')                    # IF COMPRESSION NOT ENABLED
            
            # Modify the units attribute of TIME variable for including timezone digit
            ncf = nc4.Dataset(ncFileInstac,'r+',format='NETCDF4_CLASSIC')
            ncf.variables['TIME'].units = 'days since 1950-01-01T00:00:00Z'
            ncf.variables['TIME'].calendar = 'standard'
            ncf.close()
        
            logger.info(ncFilenameInstac + ' radial netCDF file succesfully created and stored in Copericus Marine Service In Situ TAC buffer (' + vers + ').')
            
        else:
            return
        
    except Exception as err:
        dmErr = True
        if 'ncFilenameInstac' in locals():
            logger.error(err.args[0] + ' in creating Copernicus Marine Service In Situ TAC radial file ' + ncFilenameInstac)
        else:
            logger.error(err.args[0] + ' in creating Copernicus Marine Service In Situ TAC radial file for timestamp ' + R.time.strftime('%Y-%m-%d %H:%M:%S'))
        return     
    
    return

def applyEHNtotalDataModel(dmTot,networkData,stationData,vers,logger):
    """
    This function applies the European standard data model to Total object and saves
    the resulting netCDF file. The Total object is also saved as .ttl file via pickle
    binary serialization.
    The function inserts information about the created netCDF file into the 
    EU HFR NODE database.
    
    INPUTS:
        dmTot: Series containing the total to be processed with the related information
        networkData: DataFrame containing the information of the network producing the total
        stationData: DataFrame containing the information of the radial sites belonging to the network
        vers: version of the data model
        logger: logger object of the current processing

        
    OUTPUTS:
        dmTot = Series containing the processed Total object with the related information
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    dmErr = False
    
    # Check if the Total was already processed
    if dmTot['NRT_processed_flag'] == 0:
    
        try:        
            # Get the Total object
            T = dmTot['Total']
            
            # Check if Total object contains data
            if T.data.empty:
                logger.info(T.file_name + ' total file is empty: EHN data model not applied')
                return dmTot
            
    #####        
    # Convert to standard data format (netCDF)  
    #####
        
            # Apply the standard data model
            T.apply_ehn_datamodel(networkData,stationData,vers)
            
            # Set the filename (with full path) for the netCDF file
            ncFilePath = buildEHNtotalFolder(networkData.iloc[0]['total_HFRnetCDF_folder_path'],T.time,vers)
            ncFilename = buildEHNtotalFilename(networkData.iloc[0]['network_id'],T.time,'.nc')
            ncFile = ncFilePath + ncFilename 
            
            # Create the destination folder
            if not os.path.isdir(ncFilePath):
                os.makedirs(ncFilePath)
            
            # Check if the netCDF file exists and remove it
            if os.path.isfile(ncFile):
                os.remove(ncFile)
            
            # Create netCDF from DataSet and save it
            T.xds.to_netcdf(ncFile, format=T.xds.attrs['netcdf_format'])            
            logger.info(ncFilename + ' total netCDF file succesfully created and stored (' + vers + ').')
            
    #####
    # Save Total object as .ttl file with pickle
    #####
    
            # Create the destination folder
            if not os.path.isdir(ncFilePath.replace('nc','ttl')):
                os.makedirs(ncFilePath.replace('nc','ttl'))
                
            # Save the ttl file
            with open(ncFile.replace('nc','ttl'), 'wb') as ttlFile:
                  pickle.dump(T, ttlFile) 
            logger.info(ncFilename.replace('nc','ttl') + ' total ttl file succesfully created and stored (' + vers + ').')
            
        except Exception as err:
            dmErr = True
            logger.error(err.args[0] + ' for total file ' + T.file_name)
            return dmTot
            
    #####
    # Update NRT_processed_flag for the processed total
    #####
        
        if not dmErr:
            # Update the local DataFrame
            dmTot['NRT_processed_flag'] = 1      
    
    return  dmTot

def applyEHNtotalQC(qcTot,networkData,vers,logger):
    """
    This function applies QC procedures to Total object according to the European 
    standard data model.
    
    INPUTS:
        qcTot: Series containing the total to be processed with the related information
        networkData: DataFrame containing the information of the network producing the total
        vers: version of the data model
        logger: logger object of the current processing

        
    OUTPUTS:
        T = processed Total object
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    qcErr = False
    
    #####        
    # Apply QC    
    #####
    
    try:      
        # Get the total object
        T = qcTot['Total']
        
        # Check if Total object contains data
        if T.data.empty:
            logger.info(T.file_name + ' total file is empty: no QC test applied')
            return T
        
        # Check if the Total was already processed
        if qcTot['NRT_processed_flag'] == 0:           
            
            # Initialize QC metadata
            T.initialize_qc()
            
            # DDNS
            T.qc_ehn_data_density_threshold(networkData.iloc[0]['total_QC_data_density_threshold'])
            
            # CSPD
            T.qc_ehn_maximum_velocity(networkData.iloc[0]['total_QC_velocity_threshold'])
            
            if T.is_wera:
                # VART
                T.qc_ehn_maximum_variance(networkData.iloc[0]['total_QC_variance_threshold'])
            else:
                # Temporal Gradient
                prevHourTime = T.time-dt.timedelta(minutes=networkData.iloc[0]['temporal_resolution'])
                prevHourBaseFolder = networkData.iloc[0]['total_HFRnetCDF_folder_path'].replace('nc','ttl')
                prevHourFolderPath = buildEHNtotalFolder(prevHourBaseFolder,prevHourTime,vers)
                prevHourFileName = buildEHNtotalFilename(networkData.iloc[0]['network_id'],prevHourTime,'.ttl')
                prevHourTotFile = prevHourFolderPath + prevHourFileName     # previous hour total file
                if os.path.exists(prevHourTotFile):
                    with open(prevHourTotFile, 'rb') as ttlFile:
                        t0 = pickle.load(ttlFile)
                else:
                    t0 = None
                T.qc_ehn_temporal_derivative(t0,networkData.iloc[0]['total_QC_temporal_derivative_threshold'])
                T.metadata['QCTest']['VART_QC'] = 'Variance Threshold QC Test not applicable to Direction Finding systems. ' + T.metadata['QCTest']['VART_QC']
            
            # GDOP
            T.qc_ehn_gdop_threshold(networkData.iloc[0]['total_QC_GDOP_threshold'])

            # Overall QC
            T.qc_ehn_overall_qc_flag()
            
            logger.info('QC tests successfully applied to total file ' + T.file_name)
        
    except Exception as err:
        qcErr = True
        logger.error(err.args[0] + ' for total file ' + T.file_name)     
    
    return T

def performRadialCombination(combRad,networkData,numActiveStations,vers,logger):
    """
    This function performs the least square combination of the input Radials and creates
    a Total object containing the resulting total current data. 
    The Total object is also saved as .ttl file via pickle binary serialization.
    The function creates a DataFrame containing the resulting Total object along with 
    related information.
    The function inserts information about the created netCDF file into the EU HFR NODE database.
    
    INPUTS:
        combRad: DataFrame containing the Radial objects to be combined with the related information
        networkData: DataFrame containing the information of the network to which the radial site belongs
        numActiveStations: number of operational radial sites
        vers: version of the data model
        logger: logger object of the current processing

        
    OUTPUTS:
        combTot = DataFrame containing the Total object obtained via the least square combination 
                  with the related information
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    crErr = False
    
    # Create the output DataFrame
    combTot = pd.DataFrame(columns=['Total', 'NRT_processed_flag'])
    
    try:
        # Check if the combination is to be performed
        if networkData.iloc[0]['radial_combination'] == 1:
            # Check if the radials were already combined
            if ((networkData.iloc[0]['network_id'] != 'HFR-WesternItaly') and (0 in combRad['NRT_combined_flag'].values)) or ((networkData.iloc[0]['network_id'] == 'HFR-WesternItaly') and (0 in combRad['NRT_processed_flag_integrated_network'].values)):
                # Get the lat/lons of the bounding box
                lonMin = networkData.iloc[0]['geospatial_lon_min']
                lonMax = networkData.iloc[0]['geospatial_lon_max']
                latMin = networkData.iloc[0]['geospatial_lat_min']
                latMax = networkData.iloc[0]['geospatial_lat_max']
                
                # Get the grid resolution in meters
                gridResolution = networkData.iloc[0]['grid_resolution'] * 1000      # Grid resolution is stored in km in the EU HFR NODE database
                
                # Create the geographical grid
                exts = combRad.extension.unique().tolist()
                if (len(exts) == 1):
                    if exts[0] == '.ruv':
                        gridGS = createLonLatGridFromBB(lonMin, lonMax, latMin, latMax, gridResolution)
                    elif exts[0] == '.crad_ascii':
                        gridGS = createLonLatGridFromBBwera(lonMin, lonMax, latMin, latMax, gridResolution)
                else:
                    gridGS = createLonLatGridFromBB(lonMin, lonMax, latMin, latMax, gridResolution)
                    
                # Scale velocities and variances of WERA radials in case of combination with CODAR radials
                if (len(exts) > 1):
                    for idx in combRad.loc[combRad['extension'] == '.crad_ascii'].loc[:]['Radial'].index:
                        combRad.loc[idx]['Radial'].data.VELO *= 100
                        combRad.loc[idx]['Radial'].data.HCSS *= 10000
                
                # Get the combination search radius in meters
                searchRadius = networkData.iloc[0]['combination_search_radius'] * 1000      # Combination search radius is stored in km in the EU HFR NODE database
                
                # Get the timestamp
                timeStamp = dt.datetime.strptime(str(combRad.iloc[0]['datetime']),'%Y-%m-%d %H:%M:%S')
                
                # Generate the combined Total
                T, warn = combineRadials(combRad,gridGS,searchRadius,gridResolution,timeStamp)
                
                # Add metadata related to bounding box
                T = addBoundingBoxMetadata(T,lonMin,lonMax,latMin,latMax,gridResolution/1000)
                
                # Update is_combined attribute
                T.is_combined = True
                
                # Add is_wera attribute
                if (len(exts) == 1):
                    if exts[0] == '.ruv':
                        T.is_wera = False
                    elif exts[0] == '.crad_ascii':
                        T.is_wera = True
                else:
                    T.is_wera = False
                
                # Add the Total object to the DataFrame
                combTot = pd.concat([combTot, pd.DataFrame([{'Total': T, 'NRT_processed_flag':0}])])
                
                if warn=='':                    
                    # Set the filename (with full path) for the ttl file
                    ttlFilePath = buildEHNtotalFolder(networkData.iloc[0]['total_HFRnetCDF_folder_path'].replace('nc','ttl'),T.time,vers)
                    ttlFilename = buildEHNtotalFilename(networkData.iloc[0]['network_id'],T.time,'.ttl')
                    ttlFile = ttlFilePath + ttlFilename 
                    
                    # Add filename and filepath to the Total object
                    T.file_path = ttlFilePath
                    T.file_name = ttlFilename
                    T.full_file = ttlFile
                    
                    # Create the destination folder
                    if not os.path.isdir(ttlFilePath):
                        os.makedirs(ttlFilePath)
                    
                    # Save Total object as .ttl file with pickle
                    with open(ttlFile, 'wb') as ttlFile:
                        pickle.dump(T, ttlFile)
                    logger.info(ttlFilename + ' combined total ttl file succesfully created and stored (' + vers + ').')
                        
                else:
                    logger.info(warn + ' for network ' + networkData.iloc[0]['network_id'] + ' at timestamp ' + timeStamp.strftime('%Y-%m-%d %H:%M:%S'))
                    return combTot            
        
    except Exception as err:
        crErr = True
        logger.error(err.args[0] + ' for network ' + networkData.iloc[0]['network_id']  + ' in radial combination at timestamp ' + timeStamp.strftime('%Y-%m-%d %H:%M:%S')) 
                 
    return combTot

def applyEHNradialDataModel(dmRad,networkData,radSiteData,vers,logger):
    """
    This function applies the European standard data model to radial object and saves
    the resulting netCDF file. The Radial object is also saved as .rdl file via pickle
    binary serialization.
    The function inserts information about the created netCDF file into the EU HFR NODE database.
    
    INPUTS:
        dmRad: Series containing the radial to be processed with the related information
        networkData: DataFrame containing the information of the network to which the radial site belongs
        radSiteData: DataFrame containing the information of the radial site that produced the radial
        vers: version of the data model
        logger: logger object of the current processing

        
    OUTPUTS:
        dmRad = Series containing the processed Radial object with the related information
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    dmErr = False
    
    # Check if the Radial was already processed
    if dmRad['NRT_processed_flag'] == 0:
    
        try:        
        
    #####
    # Enhance Radial object with information from EU HFR NODE database
    #####
        
            # Get the Radial object
            R = dmRad['Radial']
            
            # Check if Radial object contains data
            if R.data.empty:
                logger.info(R.file_name + ' radial file is empty: EHN data model not applied')
                return dmRad
            
            # Add metadata related to range limits
            R.metadata['RangeMin'] = '0 km'
            if not R.is_wera:
                if 'RangeResolutionKMeters' in R.metadata:
                    R.metadata['RangeMax'] = str(float(R.metadata['RangeResolutionKMeters'].split()[0])*(radSiteData.iloc[0]['number_of_range_cells']-1)) + ' km'
                elif 'RangeResolutionMeters' in R.metadata:
                    R.metadata['RangeMax'] = str((float(R.metadata['RangeResolutionMeters'].split()[0]) * 0.001)*(radSiteData.iloc[0]['number_of_range_cells']-1)) + ' km'
            else:
                R.metadata['RangeMax'] = str(float(R.metadata['Range'].split()[0])*(radSiteData.iloc[0]['number_of_range_cells']-1)) + ' km'
            
    #####        
    # Convert to standard data format (netCDF)  
    #####
        
            # Apply the standard data model
            R.apply_ehn_datamodel(networkData,radSiteData,vers)
            
            # Set the filename (with full path) for the netCDF file
            ncFilePath = buildEHNradialFolder(radSiteData.iloc[0]['radial_HFRnetCDF_folder_path'],radSiteData.iloc[0]['station_id'],R.time,vers)
            ncFilename = buildEHNradialFilename(radSiteData.iloc[0]['network_id'],radSiteData.iloc[0]['station_id'],R.time,'.nc')
            ncFile = ncFilePath + ncFilename 
            
            # Create the destination folder
            if not os.path.isdir(ncFilePath):
                os.makedirs(ncFilePath)
            
            # Check if the netCDF file exists and remove it
            if os.path.isfile(ncFile):
                os.remove(ncFile)
            
            # Create netCDF from DataSet and save it
            R.xds.to_netcdf(ncFile, format=R.xds.attrs['netcdf_format'])            
            logger.info(ncFilename + ' radial netCDF file succesfully created and stored (' + vers + ').')
            
    #####
    # Save Radial object as .rdl file with pickle
    #####
    
            # Create the destination folder
            if not os.path.isdir(ncFilePath.replace('nc','rdl')):
                os.makedirs(ncFilePath.replace('nc','rdl'))
                
            # Save the rdl file
            with open(ncFile.replace('nc','rdl'), 'wb') as rdlFile:
                  pickle.dump(R, rdlFile) 
            logger.info(ncFilename.replace('nc','rdl') + ' radial rdl file succesfully created and stored (' + vers + ').')
            
        except Exception as err:
            dmErr = True
            logger.error(err.args[0] + ' for radial file ' + R.file_name)
            return dmRad
            
    #####
    # Update NRT_processed_flag for the processed radial
    #####
        
        if not dmErr:
            # Update the local DataFrame
            dmRad['NRT_processed_flag'] = 1      
    
    return dmRad

def applyEHNradialQC(qcRad,radSiteData,vers,logger):
    """
    This function applies QC procedures to radial object according to the European 
    standard data model.
    
    INPUTS:
        qcRad: Series containing the radial to be processed with the related information
        radSiteData: DataFrame containing the information of the radial site that produced the radial
        vers: version of the data model
        logger: logger object of the current processing

        
    OUTPUTS:
        R = processed Radial object
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    qcErr = False
    
    #####        
    # Apply QC    
    #####
    
    try:      
        # Get the radial object
        R = qcRad['Radial']
        
        # Check if Radial object contains data
        if R.data.empty:
            logger.info(R.file_name + ' radial file is empty: no QC test applied')
            return R
        
        # Check if the Radial was already processed
        if qcRad['NRT_processed_flag'] == 0:           
            
            # Initialize QC metadata
            R.initialize_qc()
            
            # OWTR
            R.qc_ehn_over_water()
            
            # CSPD
            R.qc_ehn_maximum_velocity(radSiteData.iloc[0]['radial_QC_velocity_threshold'])
            
            if R.is_wera:
                # VART
                R.qc_ehn_maximum_variance(radSiteData.iloc[0]['radial_QC_variance_threshold'])
            else:
                # Temporal Gradient
                prevHourTime = R.time-dt.timedelta(minutes=radSiteData.iloc[0]['temporal_resolution'])
                prevHourBaseFolder = radSiteData.iloc[0]['radial_HFRnetCDF_folder_path'].replace('nc','rdl')
                prevHourFolderPath = buildEHNradialFolder(prevHourBaseFolder,radSiteData.iloc[0]['station_id'],prevHourTime,vers)
                prevHourFileName = buildEHNradialFilename(radSiteData.iloc[0]['network_id'],radSiteData.iloc[0]['station_id'],prevHourTime,'.rdl')
                prevHourRadFile = prevHourFolderPath + prevHourFileName     # previous hour radial file
                if os.path.exists(prevHourRadFile):
                    with open(prevHourRadFile, 'rb') as rdlFile:
                        r0 = pickle.load(rdlFile)
                else:
                    r0 = None
                R.qc_ehn_temporal_derivative(r0,radSiteData.iloc[0]['radial_QC_temporal_derivative_threshold'])
                R.metadata['QCTest']['VART_QC'] = 'Variance Threshold QC Test not applicable to Direction Finding systems. ' + R.metadata['QCTest']['VART_QC']
            
            # MDFL
            R.qc_ehn_median_filter(radSiteData.iloc[0]['radial_QC_median_filter_RCLim'],radSiteData.iloc[0]['radial_QC_median_filter_CurLim'])

            # AVRB
            R.qc_ehn_avg_radial_bearing(radSiteData.iloc[0]['radial_QC_average_radial_bearing_min'],radSiteData.iloc[0]['radial_QC_average_radial_bearing_max'])

            # RDCT
            R.qc_ehn_radial_count(radSiteData.iloc[0]['radial_QC_radial_count_threshold'])           

            # Overall QC
            R.qc_ehn_overall_qc_flag()
            
            logger.info('QC tests successfully applied to radial file ' + R.file_name)
        
    except Exception as err:
        qcErr = True
        logger.error(err.args[0] + ' for radial file ' + R.file_name)     
    
    return  R 

def processTotals(dfTot,networkID,networkData,stationData,instacFolder,vers,logger):
    """
    This function processes the input total files pushed by the HFR data providers 
    according to the workflow of the EU HFR NODE.
    QC is applied to totals and they are then converted into the European standard 
    data model.
    Information about total processing is inserted into the EU HFR NODE EU HFR NODE database.
    
    INPUTS:
        dfTot: DataFrame containing the totals to be processed grouped by timestamp
                    for the input network with the related information
        networkID: network ID of the network to be processed
        networkData: DataFrame containing the information of the network to be processed
        stationData: DataFrame containing the information of the stations belonging 
                     to the network to be processed
        instacFolder: full path of the folder where to save data for Copernicus Marine Service 
                    (if None, no files for Copernicus Marine Service are produced)
        vers: version of the data model
        logger: logger object of the current processing

        
    OUTPUTS:
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    pTerr = False
    
    try:
                
        #####
        # Enhance the total DataFrame
        #####
        
        # Add Total objects to the DataFrame
        if 'HFR-US' in networkID:
            dfTot['Total'] = (dfTot.filepath + '/' + dfTot.filename).apply(lambda x: pickle.load(open(x, 'rb')))
        else:
            dfTot['Total'] = (dfTot.filepath + '/' + dfTot.filename).apply(lambda x: Total(x))
        
        # Add metadata related to bounding box
        lonMin = networkData.iloc[0]['geospatial_lon_min']
        lonMax = networkData.iloc[0]['geospatial_lon_max']
        latMin = networkData.iloc[0]['geospatial_lat_min']
        latMax = networkData.iloc[0]['geospatial_lat_max']
        gridRes = networkData.iloc[0]['grid_resolution']
        dfTot['Total'] = dfTot['Total'].apply(lambda x: addBoundingBoxMetadata(x,lonMin,lonMax,latMin,latMax,gridRes))
        
        #####
        # Manage site codes for WERA networks
        #####
        
        # HFR-NADr
        if networkID == 'HFR-NAdr':
            dfTot.iloc[0]['Total'].site_source['Name']=dfTot.iloc[0]['Total'].site_source['Name'].str.replace('Izola','IZOL')
            dfTot.iloc[0]['Total'].site_source['Name']=dfTot.iloc[0]['Total'].site_source['Name'].str.replace('Trieste Dam','TRI1')
            dfTot.iloc[0]['Total'].site_source['Name']=dfTot.iloc[0]['Total'].site_source['Name'].str.replace('Slovenia1','PIRA')
            dfTot.iloc[0]['Total'].site_source['Name']=dfTot.iloc[0]['Total'].site_source['Name'].str.replace('Trieste','AURI')
            
        # HFR-COSYNA
        if networkID == 'HFR-COSYNA':
            dfTot.iloc[0]['Total'].site_source['Name']=dfTot.iloc[0]['Total'].site_source['Name'].str.replace('Buesum','BUES')
            dfTot.iloc[0]['Total'].site_source['Name']=dfTot.iloc[0]['Total'].site_source['Name'].str.replace('Sylt','SYLT')
            dfTot.iloc[0]['Total'].site_source['Name']=dfTot.iloc[0]['Total'].site_source['Name'].str.replace('Wangerooge','WANG')
        
        #####        
        # Apply QC to Totals
        #####
        
        dfTot['Total'] = dfTot.apply(lambda x: applyEHNtotalQC(x,networkData,vers,logger),axis=1)
        
        #####        
        # Convert Total to standard data format (netCDF)
        #####
        
        # European standard data model
        dfTot = dfTot.apply(lambda x: applyEHNtotalDataModel(x,networkData,stationData,vers,logger),axis=1)
        
        if instacFolder:
            # Copernicus Marine Service In Situ TAC data model
            dfTot = dfTot.apply(lambda x: applyINSTACtotalDataModel(x,networkData,stationData,instacFolder,vers,logger),axis=1)
        
    except Exception as err:
        pTerr = True
        logger.error(err.args[0])    
    
    return
    
def processRadials(groupedRad,networkID,networkData,stationData,instacFolder,vers,logger):
    """
    This function processes the input radial files pushed by the HFR data providers 
    according to the workflow of the EU HFR NODE.
    QC is applied to radials and they are then converted into the European standard 
    data model.
    If the radial combination is enabled, radials are combined into totals, 
    QC is applied to the resulting totals and these are then converted into the European 
    standard data model.
    Information about radial and total processing is inserted into the EU HFR NODE database.
    
    INPUTS:
        groupedRad: DataFrame containing the radials to be processed grouped by timestamp
                    for the input network with the related information
        networkID: network ID of the network to be processed
        networkData: DataFrame containing the information of the network to be processed
        stationData: DataFrame containing the information of the stations belonging 
                     to the network to be processed
        instacFolder: full path of the folder where to save data for Copernicus Marine Service 
                    (if None, no files for Copernicus Marine Service are produced)
        vers: version of the data model
        logger: logger object of the current processing

        
    OUTPUTS:
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    pRerr = False
    
    try:
        
        #####
        # Enhance the radial DataFrame
        #####
        
        # Add Radial objects to the DataFrame
        groupedRad['Radial'] = (groupedRad.filepath + '/' + groupedRad.filename).apply(lambda x: Radial(x))
        
        # Add metadata related to bounding box
        lonMin = networkData.iloc[0]['geospatial_lon_min']
        lonMax = networkData.iloc[0]['geospatial_lon_max']
        latMin = networkData.iloc[0]['geospatial_lat_min']
        latMax = networkData.iloc[0]['geospatial_lat_max']
        gridRes = networkData.iloc[0]['grid_resolution']
        groupedRad['Radial'] = groupedRad['Radial'].apply(lambda x: addBoundingBoxMetadata(x,lonMin,lonMax,latMin,latMax,gridRes))
        
        # Rename indices with site codes
        indexMapper = dict(zip(groupedRad.index.values.tolist(),groupedRad['station_id'].to_list()))
        groupedRad.rename(index=indexMapper,inplace=True)    
        
        if networkID != 'HFR-WesternItaly':
        
        #####        
        # Apply QC to Radials
        #####
        
            groupedRad['Radial'] = groupedRad.apply(lambda x: applyEHNradialQC(x,stationData.loc[stationData['station_id'] == x.station_id],vers,logger),axis=1)
        
        #####        
        # Convert Radials to standard data format (netCDF)
        #####
        
            # European standard data model
            groupedRad = groupedRad.apply(lambda x: applyEHNradialDataModel(x,networkData,stationData.loc[stationData['station_id'] == x.station_id],vers,logger),axis=1)
            
            if instacFolder:
                # Copernicus Marine Service In Situ TAC data model
                groupedRad.apply(lambda x: applyINSTACradialDataModel(x,networkData,stationData.loc[stationData['station_id'] == x.station_id],instacFolder,vers,logger),axis=1)
                
        #####
        # Combine Radials into Total
        #####
        
        # Check if the combination is to be performed
        if networkData.iloc[0]['radial_combination'] == 1:
            # Check if at least two Radials are available for combination
            if len(groupedRad) > 1:
                dfTot = performRadialCombination(groupedRad,networkData,vers,logger)                
            
            if 'dfTot' in locals():
            
            #####        
            # Apply QC to Totals
            #####
            
                dfTot['Total'] = dfTot.apply(lambda x: applyEHNtotalQC(x,networkData,vers,logger),axis=1)        
                
            #####        
            # Convert Totals to standard data format (netCDF)
            #####
            
                # European standard data model
                dfTot = dfTot.apply(lambda x: applyEHNtotalDataModel(x,networkData,stationData,vers,logger),axis=1)
                
                if instacFolder:
                    # Copernicus Marine Service In Situ TAC data model
                    dfTot = dfTot.apply(lambda x: applyINSTACtotalDataModel(x,networkData,stationData,instacFolder,vers,logger),axis=1)
        
    except Exception as err:
        pRerr = True
        logger.error(err.args[0])    
    
    return

def selectUStotals(networkID,networkData,stationData,startDate,endDate,vers,logger):
    """
    This function reads data from HFR US network via OpenDAP, selects the data subset to be processed
    according to the processing time interval, produces and stores hourly Total objects from hourly 
    subsets of the selected data and creates the DataFrame containing the information needed for the 
    generation of the total data files into the European standard data model.
    
    INPUTS:
        networkID: network ID of the network to be processed
        networkData: DataFrame containing the information of the network to be processed
        stationData: DataFrame containing the information of the stations belonging 
                     to the network to be processed
        startDate: datetime of the initial date of the processing period
        endDate: datetime of the final date of the processing period
        vers: version of the data model
        logger: logger object of the current processing

        
    OUTPUTS:
        totalsToBeProcessed: DataFrame containing all the totals to be processed for the input 
                              network with the related information
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    sTerr = False
    
    # Create output total Series
    totalsToBeProcessed = pd.DataFrame(columns=['filename', 'filepath', 'network_id', 'timestamp', 'datetime', 'reception_date', \
                                                'filesize', 'extension', 'NRT_processed_flag'])
    
    #####
    # Load totals from the TDS catalog
    #####
    
    try:
        logger.info('Total input started for ' + networkID + ' network.')
        
        # Trim heading and trailing whitespaces from TDS root URL
        TDSrootURL = networkData.iloc[0]['TDS_root_url'].strip()
        
        # Load data (xarray initially loads only the information on the data and not the values)
        UStdsDS=xr.open_dataset(TDSrootURL,decode_times=True)        
        
    #####
    # Select the total data by checking the timestamps
    #####
    
        # Find the indices of the dataset timestamps in the time interval to be processed
        idxToBeInserted = np.searchsorted(UStdsDS['time'].to_numpy(),UStdsDS['time'].where((UStdsDS.time>=startDate and UStdsDS.time<=endDate), drop=True).to_numpy())
        
        # Select data for the timestamps to be inserted
        USxds = UStdsDS.isel(time=idxToBeInserted)
    
    #####
    # Create and store Total objects for each timestamp
    #####
    
        # Create a pandas DataFrame containing the timestamps to be processed
        tsDF = pd.DataFrame(data=USxds.time.values,columns=['timestamp'])
    
        # Get longitude and latitude values of the input data geographical grid
        lonDim = USxds.lon.to_numpy()
        latDim = USxds.lat.to_numpy()
        
        # Get the longitude/latitude couples
        Lon, Lat = np.meshgrid(lonDim, latDim)
        # Create grid
        Lonc = Lon.flatten()
        Latc = Lat.flatten()
    
        # Now convert these points to geo-data
        positions = gpd.GeoSeries([Point(x, y) for x, y in zip(Lonc, Latc)])
        positions = positions.set_crs('epsg:4326')    
    
        # Create and save the Total objects
        tsDF['timestamp'].apply(lambda x: createTotalFromUStds(x,positions,USxds.where(USxds.time == x, drop=True),networkData,stationData,vers,logger))   
    
    #####
    # List totals from the network
    #####
    
        # Build input folder path string
        inputFolder = networkData.iloc[0]['total_HFRnetCDF_folder_path'].strip().replace('nc','ttl')       
        # Check if the input folder is specified
        if(not inputFolder):
            logger.info('No total input folder specified for network ' + networkID)
        else:
            # Check if the input folder path exists
            if not os.path.isdir(inputFolder):
                logger.info('The total input folder for network ' + networkID + ' does not exist.')
            else:
                # Consider file type for totals from US networks
                usTypeWildcard = '**/*.ttl'
                # List all total files
                inputFiles = [file for file in glob.glob(os.path.join(inputFolder,usTypeWildcard), recursive = True)]                    
                for inputFile in inputFiles:
                    try:
                        # Get file parts
                        filePath = os.path.dirname(inputFile)
                        fileName = os.path.basename(inputFile)
                        fileExt = os.path.splitext(inputFile)[1]
                        
                        # Get file timestamp
                        with open(inputFile, 'rb') as ttlFile:
                            total = pickle.load(ttlFile)
                        timeStamp = total.time.strftime("%Y %m %d %H %M %S")                    
                        dateTime = total.time.strftime("%Y-%m-%d %H:%M:%S")  
                        
                        # Get file size in Kbytes
                        fileSize = os.path.getsize(inputFile)/1024   
                                
    #####
    # Insert total information into the output DataFrame
    #####
    
                        # Check if the radial falls into the processing time interval
                        if ((total.time >= startDate) and (total.time <= endDate)):
                            
                            # Prepare data to be inserted into the output DataFrame
                            dataTotal = {'filename': [fileName], 'filepath': [filePath], 'network_id': [networkID], 'timestamp': [timeStamp], \
                                     'datetime': [dateTime], 'reception_date': [dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")], \
                                      'filesize': [fileSize], 'extension': [fileExt], 'NRT_processed_flag': [0]}
                            dfTotal = pd.DataFrame(dataTotal)
                            
                            # Insert into the output DataFrame
                            totalsToBeProcessed = pd.concat([totalsToBeProcessed, dfTotal])
                                
                    except Exception as err:
                        sTerr = True
                        logger.error(err.args[0] + ' for file ' + fileName)
                    
    except Exception as err:
        sTerr = True
        logger.error(err.args[0])
    
    return totalsToBeProcessed

def selectTotals(networkID,networkData,startDate,endDate,logger):
    """
    This function lists the input total files pushed by the HFR data providers 
    that falls into the processing time interval and creates the DataFrame containing 
    the information needed for the generation of the total data files into the 
    European standard data model.
    
    INPUTS:
        networkID: network ID of the network to be processed
        networkData: DataFrame containing the information of the network to be processed
        startDate: datetime of the initial date of the processing period
        endDate: datetime of the final date of the processing period
        logger: logger object of the current processing

        
    OUTPUTS:
        totalsToBeProcessed: DataFrame containing all the totals to be processed for the input 
                              network with the related information
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    sTerr = False
    
    # Create output total Series
    totalsToBeProcessed = pd.DataFrame(columns=['filename', 'filepath', 'network_id', 'timestamp', 'datetime', 'reception_date', \
                                                'filesize', 'extension', 'NRT_processed_flag'])
    
    #####
    # List totals from the network
    #####
    
    try:
        logger.info('Total input started for ' + networkID + ' network.')
        # Trim heading and trailing whitespaces from input folder path string
        inputFolder = networkData.iloc[0]['total_input_folder_path'].strip()
        # Check if the input folder is specified
        if(not inputFolder):
            logger.info('No total input folder specified for network ' + networkID)
        else:
            # Check if the input folder path exists
            if not os.path.isdir(inputFolder):
                logger.info('The total input folder for network ' + networkID + ' does not exist.')
            else:
                # Consider file types for Codar and WERA systems
                codarTypeWildcard = '**/*.tuv'      # Codar systems
                weraTypeWildcard = '**/*.cur_asc'   # WERA systems                
                # List all total files
                codarInputFiles = [file for file in glob.glob(os.path.join(inputFolder,codarTypeWildcard), recursive = True)]                    
                weraInputFiles = [file for file in glob.glob(os.path.join(inputFolder,weraTypeWildcard), recursive = True)]
                inputFiles = codarInputFiles + weraInputFiles
                for inputFile in inputFiles:
                    try:
                        # Get file parts
                        filePath = os.path.dirname(inputFile)
                        fileName = os.path.basename(inputFile)
                        fileExt = os.path.splitext(inputFile)[1]
                        
                        # Get file timestamp
                        total = Total(inputFile)
                        timeStamp = total.time.strftime("%Y %m %d %H %M %S")                    
                        dateTime = total.time.strftime("%Y-%m-%d %H:%M:%S")  
                        
                        # Get file size in Kbytes
                        fileSize = os.path.getsize(inputFile)/1024   
                        
    #####
    # Insert total information into the output DataFrame
    #####
    
                        # Check if the radial falls into the processing time interval
                        if ((total.time >= startDate) and (total.time <= endDate)):

                            # Prepare data to be inserted into the output DataFrame
                            dataTotal = {'filename': [fileName], 'filepath': [filePath], 'network_id': [networkID], 'timestamp': [timeStamp], \
                                         'datetime': [dateTime], 'reception_date': [dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")], \
                                          'filesize': [fileSize], 'extension': [fileExt], 'NRT_processed_flag': [0]}
                            dfTotal = pd.DataFrame(dataTotal)
                                
                            # Insert into the output DataFrame
                            totalsToBeProcessed = pd.concat([totalsToBeProcessed, dfTotal])
                                
                    except Exception as err:
                        sTerr = True
                        logger.error(err.args[0] + ' for file ' + fileName)
                    
    except Exception as err:
        sTerr = True
        logger.error(err.args[0])
    
    return totalsToBeProcessed

def selectRadials(networkID,stationData,startDate,endDate,logger):
    """
    This function lists the input radial files pushed by the HFR data providers 
    that falls into the processing time interval and creates the DataFrame containing 
    the information needed for the combination of radial files into totals and for the
    generation of the radial and total data files into the European standard data model.
    
    INPUTS:
        networkID: network ID of the network to be processed
        stationData: DataFrame containing the information of the stations belonging 
                     to the network to be processed
        startDate: datetime of the initial date of the processing period
        endDate: datetime of the final date of the processing period
        logger: logger object of the current processing

        
    OUTPUTS:
        radialsToBeProcessed: DataFrame containing all the radials to be processed for the input 
                              network with the related information
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    sRerr = False
    
    # Create output total Series
    radialsToBeProcessed = pd.DataFrame(columns=['filename', 'filepath', 'network_id', 'station_id', \
                                                 'timestamp', 'datetime', 'reception_date', 'filesize', 'extension', \
                                                 'NRT_processed_flag', 'NRT_processed_flag_integrated_network', 'NRT_combined_flag'])
    
    #####
    # List radials from stations
    #####
    
    # Scan stations
    for st in range(len(stationData)):
        try:   
            # Get station id
            stationID = stationData.iloc[st]['station_id']
            logger.info('Radial input started for ' + networkID + '-' + stationID + ' station.')
            # Trim heading and trailing whitespaces from input folder path string
            inputFolder = stationData.iloc[st]['radial_input_folder_path'].strip()
            # Check if the input folder is specified
            if(not inputFolder):
                logger.info('No radial input folder specified for station ' + networkID + '-' + stationID)
            else:
                # Check if the input folder path exists
                if not os.path.isdir(inputFolder):
                    logger.info('The radial input folder for station ' + networkID + '-' + stationID + ' does not exist.')
                else:
                    # Get the input file type (based on manufacturer)
                    manufacturer = stationData.iloc[st]['manufacturer'].lower()
                    if 'codar' in manufacturer:
                        fileTypeWildcard = '**/*.ruv'
                    elif 'wera' in manufacturer:
                        fileTypeWildcard = '**/*.crad_ascii'                
                    # List all radial files
                    inputFiles = [file for file in glob.glob(os.path.join(inputFolder,fileTypeWildcard), recursive = True)]                    
                    for inputFile in inputFiles:
                        try:
                            # Get file parts
                            filePath = os.path.dirname(inputFile)
                            fileName = os.path.basename(inputFile)
                            fileExt = os.path.splitext(inputFile)[1]
                            
                            # Get file timestamp
                            radial = Radial(inputFile)
                            timeStamp = radial.time.strftime("%Y %m %d %H %M %S")                    
                            dateTime = radial.time.strftime("%Y-%m-%d %H:%M:%S")  
                            
                            # Get file size in Kbytes
                            fileSize = os.path.getsize(inputFile)/1024 
                            
    #####
    # Insert radial information into the output DataFrame
    #####
    
                            # Check if the radial falls into the processing time interval
                            if ((radial.time >= startDate) and (radial.time <= endDate)):
    
                                # Prepare data to be inserted into the output DataFrame
                                dataRadial = {'filename': [fileName], 'filepath': [filePath], 'network_id': [networkID], \
                                              'station_id': [stationID], 'timestamp': [timeStamp], 'datetime': [dateTime], \
                                              'reception_date': [dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")], \
                                              'filesize': [fileSize], 'extension': [fileExt], 'NRT_processed_flag': [0], \
                                              'NRT_processed_flag_integrated_network': [0], 'NRT_combined_flag': [0]}
                                dfRadial = pd.DataFrame(dataRadial)
                                
                                # Insert into the output DataFrame
                                radialsToBeProcessed = pd.concat([radialsToBeProcessed, dfRadial])

                        except Exception as err:
                            sRerr = True
                            logger.error(err.args[0] + ' for file ' + fileName)
                        
        except Exception as err:
            sRerr = True
            logger.error(err.args[0] + ' for station ' + stationID)
    
    return radialsToBeProcessed

def processNetwork(networkID,startDate,endDate,dataFolder,instacFolder,sqlConfig):
    """
    This function processes the radial and total files of a single HFR network
    for generating radial and total files according to the European standard data model.
    
    The first processing step consists in the listing of the input files
    (both radial and total) pushed by the HFR data providers.
    
    The second processing step consists in reading the EU HFR NODE EU HFR NODE database for collecting
    information about the radial data files to be combined into totals and in
    combining and generating radial and total data files according to the European
    standard data model and to the Copernicus Marine Service In-Situ TAC data model.
    
    The third processing step consists in reading the EU HFR NODE EU HFR NODE database for collecting
    information about the total data files to be converted into the European standard
    data model and in the generating total data files according to the European
    standard data model and to the Copernicus Marine Service In-Situ TAC data model.
    
    INPUTS:
        networkID: network ID of the network to be processed
        startDate: initial date of the processing period in datetime format
        endDate: final date of the processing period in datetime format
        dataFolder: full path of the folder containing network data 
                    (if None, data folder paths read from the database are used)
        instacFolder: full path of the folder where to save data for Copernicus Marine Service 
                    (if None, no files for Copernicus Marine Service are produced)
        sqlConfig: parameters for connecting to the Mysql EU HFR NODE EU HFR NODE database

        
    OUTPUTS:
        pNerr: error flag (True = errors occurred, False = no error occurred)
        
    """
    #####
    # Setup
    #####
    
    # Set the version of the data model
    vers = 'v3'
    
    try:
        # Create the folder for the network log
        networkLogFolder = '/var/log/EU_HFR_NODE_HIST/' + networkID
        if not os.path.isdir(networkLogFolder):
            os.mkdir(networkLogFolder)
               
        # Create logger
        logger = logging.getLogger('EU_HFR_NODE_HIST_' + networkID)
        logger.setLevel(logging.INFO)
        # Create console handler and set level to DEBUG
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # Create logfile handler
        lfh = logging.FileHandler(networkLogFolder + '/EU_HFR_NODE_HIST_' + networkID + '.log')
        lfh.setLevel(logging.INFO)
        # Create formatter
        formatter = logging.Formatter('[%(asctime)s] -- %(levelname)s -- %(module)s - %(funcName)s - %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
        # Add formatter to lfh and ch
        lfh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # Add lfh and ch to logger
        logger.addHandler(lfh)
        logger.addHandler(ch)
        
        # Initialize error flag
        pNerr = False
        
    except Exception as err:
        pNerr = True
        logger.error(err.args[0])
        logger.info('Exited with errors.')
        return pNerr
    
    #####
    # Retrieve information about network and stations from EU HFR NODE database
    #####
    
    try:
        # Create SQLAlchemy engine for connecting to EU HFR NODE database
        eng = sqlalchemy.create_engine('mysql+mysqlconnector://' + sqlConfig['user'] + ':' + \
                                       sqlConfig['password'] + '@' + sqlConfig['host'] + '/' + \
                                       sqlConfig['EU HFR NODE database'])
        
        # Set and execute the query and get the HFR network data
        networkSelectQuery = 'SELECT * FROM network_tb WHERE network_id=\'' + networkID + '\''
        networkData = pd.read_sql(networkSelectQuery, con=eng)
        numNetworks = networkData.shape[0]
        logger.info(networkID + ' network data successfully fetched from EU HFR NODE database.')
        # Set and execute the query and get the HFR station data
        if networkID == 'HFR-WesternItaly':
            stationSelectQuery = 'SELECT * FROM station_tb WHERE network_id=\'HFR-TirLig\' OR network_id=\'HFR-LaMMA\' OR network_id=\'HFR-ARPAS\''
        else:
            stationSelectQuery = 'SELECT * FROM station_tb WHERE network_id=\'' + networkID + '\''
        stationData = pd.read_sql(stationSelectQuery, con=eng)
        numStations = stationData.shape[0]
        
        # Avoid to have None in the last_Calibration_date field
        stationData['last_calibration_date']=stationData['last_calibration_date'].apply(lambda x: dt.date(1,1,1) if x is None else x)
        
        logger.info(networkID + ' station data successfully fetched from EU HFR NODE database.')
    except sqlalchemy.exc.DBAPIError as err:        
        pNerr = True
        logger.error('MySQL error ' + err._message())
        logger.info('Exited with errors.')
        return pNerr
        
    #####
    # Select HFR data
    #####
    
    try:
        # Modify data folders (if needed)
        if dataFolder:
            # Modify total data folder paths
            networkData = networkData.apply(lambda x: modifyNetworkDataFolders(x,dataFolder,logger),axis=1)
            # Modify radial data folder paths
            stationData = stationData.apply(lambda x: modifyStationDataFolders(x,dataFolder,logger),axis=1)
            
            
        # Select radials to be processed
        if 'HFR-US' in networkID:
            pass
        else:
            radialsToBeProcessed = selectRadials(networkID, stationData, startDate, endDate, logger)
            logger.info('Radials to be processed successfully selected for network ' + networkID)
        
        # Select totals to be processed
        if 'HFR-US' in networkID:
            totalsToBeProcessed = selectUStotals(networkID, networkData, stationData, startDate, vers, logger)
        elif networkID == 'HFR-WesternItaly':
            pass
        else:
            if networkData.iloc[0]['radial_combination'] == 0:
                totalsToBeProcessed = selectTotals(networkID, networkData, startDate, logger)
                logger.info('Totals to be processed successfully selected for network ' + networkID)
        
    #####
    # Process HFR data
    #####
        
        # Process radials
        if 'HFR-US' in networkID:
            pass
        else:            
            logger.info('Radial processing started for ' + networkID + ' network') 
            radialsToBeProcessed.groupby('datetime', group_keys=False).apply(lambda x:processRadials(x,networkID,networkData,stationData,instacFolder,vers,eng,logger))
        
        # Process totals
            logger.info('Total processing started for ' + networkID + ' network') 
            totalsToBeProcessed.groupby('datetime', group_keys=False).apply(lambda x:processTotals(x,networkID,networkData,stationData,instacFolder,vers,eng,logger))
            
        # Wait a bit (useful for multiprocessing management)
        time.sleep(30)
            
    except Exception as err:
        pNerr = True
        logger.error(err.args[0])
        logger.info('Exited with errors.')
        return pNerr    
    
    return pNerr

####################
# MAIN DEFINITION
####################

def main(argv):
    
#####
# Setup
#####
       
    # Set the argument structure
    try:
        opts, args = getopt.getopt(argv,"n:s:e:d:i:h",["network=","start-date=","end-date=","data-folder=","instac-folder=","help"])
    except getopt.GetoptError:
        print('Usage: EU_HFR_NODE_HISTprocessor.py -n <network ID of the network to be processed (if not specified, all the networks are processed)> ' \
              + '-s <initial date for processing formatted as yyyy-mm-dd (ISO8601 UTC date representation)> ' \
                  + '-e <final date for processing formatted as yyyy-mm-dd (ISO8601 UTC date representation)> ' \
                      + '-d <full path of the folder containing network data (if not specified, data folder paths read from the database are used)> ' \
                          + '-i <full path of the folder where to save data for Copernicus Marine Service (if not specified, no files for Copernicus Marine Service are produced)>')
        sys.exit(2)
        
    if not argv:
        print("No processing time interval specified. Please type 'EU_HFR_NODE_HISTprocessor.py -h' for help.")
        sys.exit(2)
        
    if (('-s' not in argv) and ('--start-date' not in argv)) or (('-e' not in argv) and ('--end-date' not in argv)):
        print('Usage: EU_HFR_NODE_HISTprocessor.py -n <network ID of the network to be processed (if not specified, all the networks are processed)> ' \
              + '-s <initial date for processing formatted as yyyy-mm-dd (ISO8601 UTC date representation)> ' \
                  + '-e <final date for processing formatted as yyyy-mm-dd (ISO8601 UTC date representation)> ' \
                      + '-d <full path of the folder containing network data (if not specified, data folder paths read from the database are used)> ' \
                          + '-i <full path of the folder where to save data for Copernicus Marine Service (if not specified, no files for Copernicus Marine Service are produced)>')
        sys.exit(2)
        
    # Initialize optional arguments
    ntw = None
    dataFolder = None
    instacFolder = None
        
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: EU_HFR_NODE_HISTprocessor.py -n <network ID of the network to be processed (if not specified, all the networks are processed)> ' \
                  + '-s <initial date for processing formatted as yyyy-mm-dd (ISO8601 UTC date representation)> ' \
                      + '-e <final date for processing formatted as yyyy-mm-dd (ISO8601 UTC date representation)> ' \
                          + '-d <full path of the folder containing network data (if not specified, data folder paths read from the database are used)> ' \
                              + '-i <full path of the folder where to save data for Copernicus Marine Service (if not specified, no files for Copernicus Marine Service are produced)>')
            sys.exit()
        elif opt in ("-n", "--network"):
            ntw = arg
        elif opt in ("-s", "--start-date"):
            # Check date format
            try:
                dateCheck = dt.datetime.strptime(arg, '%Y-%m-%d')
                startDate = dateCheck
            except ValueError:
                print("Incorrect format for initial date, should be yyyy-mm-dd (i.e. ISO8601 UTC date representation)")
                sys.exit(2)
        elif opt in ("-e", "--end-date"):
            # Check date format
            try:
                dateCheck = dt.datetime.strptime(arg, '%Y-%m-%d')
                endDate = dateCheck
            except ValueError:
                print("Incorrect format for final date, should be yyyy-mm-dd (i.e. ISO8601 UTC date representation)")
                sys.exit(2)
        elif opt in ("-d", "--data-folder"):
            dataFolder = arg.strip()
            # Check if the data folder path exists
            if not os.path.isdir(dataFolder):
                print('The specified data folder does not exist.')
                sys.exit(2)
        elif opt in ("-i", "--instac-folder"):
            instacFolder = arg.strip()
            # Check if the INSTAC folder path exists
            if not os.path.isdir(instacFolder):
                print('The specified folder for Copernicus Marine Service data does not exist.')
                sys.exit(2)
            
    # Check that initial date is before end date
    if not startDate<endDate:
        print("Wrong time interval specified: initial date is later then end date")
        sys.exit(2)
          
    # Create logger
    logger = logging.getLogger('EU_HFR_NODE_HIST')
    logger.setLevel(logging.INFO)
    # Create console handler and set level to DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # Create logfile handler
    lfh = logging.FileHandler('/var/log/EU_HFR_NODE_HIST/EU_HFR_NODE_HIST.log')
    lfh.setLevel(logging.INFO)
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] -- %(levelname)s -- %(module)s - %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    # Add formatter to lfh and ch
    lfh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # Add lfh and ch to logger
    logger.addHandler(lfh)
    logger.addHandler(ch)
    
    # Set parameter for Mysql EU HFR NODE database connection
    sqlConfig = {
      'user': 'HFRuserCP',
      'password': '!_kRIVAHYH2RLpmQxz_!',
      'host': '150.145.136.104',
      'EU HFR NODE database': 'HFR_node_db',
    }
    
    # Initialize error flag
    EHNerr = False
    
    logger.info('Historical processing started.')
    
#####
# Network data collection
#####
    
    # Check if a specific network is selected for processing (if not, get all network IDs)
    if not ntw:
        try:
            # Create SQLAlchemy engine for connecting to EU HFR NODE database
            eng = sqlalchemy.create_engine('mysql+mysqlconnector://' + sqlConfig['user'] + ':' + \
                                           sqlConfig['password'] + '@' + sqlConfig['host'] + '/' + \
                                           sqlConfig['EU HFR NODE database'])
                
            # Set and execute the query and get the HFR network IDs to be processed
            networkSelectQuery = 'SELECT network_id FROM network_tb WHERE EU_HFR_processing_flag=1'
            networkIDs = pd.read_sql(networkSelectQuery, con=eng)['network_id'].to_list()
            logger.info('Network IDs successfully fetched from EU HFR NODE database.')
        except sqlalchemy.exc.DBAPIError as err:        
            EHNerr = True
            logger.error('MySQL error ' + err._message())
            logger.info('Exited with errors.')
            sys.exit()
    
#####
# Process launch and monitor
#####

    try:
        # Check if a specific network is selected for processing
        if ntw:
            processNetwork(ntw, startDate, endDate, dataFolder, instacFolder, sqlConfig)
        # Process all networks if no specific network is specified for processing
        else:
            # Set the queue containing the network IDs
            networkQueue = networkIDs
        
            # Set the batch size
            numCPU = os.cpu_count()
            if len(networkIDs) < numCPU -1:
                batchDim = len(networkIDs)
            else:
                batchDim = numCPU - 2
            
            # Start a process per each network in the batch
            prcs = {}       # dictionary of the running processes contanig processes and related network IDs
            for ntw in networkQueue[0:batchDim]:
                # Wait a bit (useful for multiprocessing management)
                time.sleep(10)
                # Start the process
                p = Process(target=processNetwork, args=(ntw, startDate, endDate, dataFolder, instacFolder, sqlConfig,))
                p.start()
                # Insert process and the related network ID into the dictionary of the running processs
                prcs[p] = ntw
                logger.info('Processing for ' + ntw + ' network started')
                
            while True:
                # check which processes are terminated and append the processed network names to the dictionary of terminated processes
                trm = {}        # dictionary of the terminated processes 
                for pp in prcs.keys():
                    if pp.exitcode != None:
                        trm[pp] = prcs[pp]
                        
                # Close the terminated processes and append them to the queue
                for tt in trm.keys():
                    # Close the process
                    tt.close()
                    prcs.pop(tt)
                    logger.info('Processing for ' + trm[tt] + ' network ended')
                    # Pop the network from the queue
                    networkQueue.remove(trm[tt])
                    # Add the tne network at the end of the queue
                    networkQueue.append(trm[tt])
                    
                    # Start a new process for the first network of the queue not in the dictionary of the running processes
                    for ntw in networkQueue:
                        if ntw not in prcs.values():
                            break
                    # Wait a bit (useful for multiprocessing management)
                    time.sleep(10)
                    # Start the process
                    p = Process(target=processNetwork, args=(ntw, startDate, endDate, dataFolder, instacFolder, sqlConfig,))
                    p.start()
                    # Insert process and the related network ID into the dictionary of the running processes
                    prcs[p] = ntw
                    logger.info('Processing for ' + ntw + ' network started')
            
    
    except Exception as err:
        EHNerr = True
        logger.error(err.args[0])
    
    
    
####################
    
    if(not EHNerr):
        logger.info('Successfully executed.')
    else:
        logger.error('Exited with errors.')
            
####################


#####################################
# SCRIPT LAUNCHER
#####################################    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    