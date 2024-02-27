#!/usr/bin/python3


# Created on Fri Feb 23 17:16:46 2024

# @author: Lorenzo Corgnati
# e-mail: lorenzo.corgnati@sp.ismar.cnr.it


# This application patches the netCDF time series of HFR surface current data produced by the EU HFR NODE
# for adding the wmo_platform_code, wigos_id and oceanops_ref global attributes to the files
# produced with the Euroepan standard data model and for filling the wmo_platform_code global attribute
# to the files produced with the Copernicus Marine Service In Situ TAC data model.

# The WMO code, the OceanOps ref and the WIGOS ID of each radial station are taken from the EU HFR NODE
# database.

# When calling the application it is possible to specify if all the networks have to be processed
# or only the selected one, and if the patching of data files data produced with the
# Copernicus Marine Service data model has to be performed.

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
from dateutil.relativedelta import relativedelta
from radials import Radial, buildEHNradialFolder, buildEHNradialFilename, convertEHNtoINSTACradialDatamodel, buildINSTACradialFolder, buildINSTACradialFilename
from totals import Total, buildEHNtotalFolder, buildEHNtotalFilename, combineRadials, convertEHNtoINSTACtotalDatamodel, buildINSTACtotalFolder, buildINSTACtotalFilename, buildUStotal
from calc import createLonLatGridFromBB, createLonLatGridFromBBwera, createLonLatGridFromTopLeftPointWera
from common import addBoundingBoxMetadata
import pickle
from concurrent import futures
import time
import xarray as xr
import netCDF4 as nc4
import shutil

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

def applyINSTACradialPatch(radFile,wmoCode,logger):
    """
    This function applies the patch to the radial netCDF file received in input
    for adding the wmo_platform_code, wigos_id and oceanops_ref global attributes 
    to the files produced with the Euroepan standard data model.
    
    INPUTS:
        radFile: Series containing the radial to be processed with the related information
        wmoCode: string containing the WMO code
        logger: logger object of the current processing

        
    OUTPUTS:
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    ipErr = False
    
    return

def applyEHNradialPatch(radFile,oceanopsRef,wmoCode,wigosId,logger):
    """
    This function applies the patch to the radial netCDF file received in input
    for adding the wmo_platform_code, wigos_id and oceanops_ref global attributes 
    to the files produced with the Euroepan standard data model.
    
    INPUTS:
        radFile: Series containing the radial to be processed with the related information
        oceanopsRef: string containing the OceanOps REF
        wmoCode: string containing the WMO code
        wigosId: string containing the WIGOS ID
        logger: logger object of the current processing

        
    OUTPUTS:
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    rpErr = False
    
    #####
    # Apply the patch for adding wmo_platform_code, ocenops_ref and wigos_id global attributes
    #####
    
    try:
        # Get the filename
        
        
        # Add wmo_platform_code, ocenops_ref and wigos:id global attributes
        ncf = nc4.Dataset(radFile,'r+',format='NETCDF4_CLASSIC')
        ncf.attrs['ocenops_ref'] = oceanopsRef
        ncf.attrs['wmo_platform_code'] = wmoCode
        ncf.attrs['wigos_id'] = wigosId
        ncf.attrs['date_modified'] = dt.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
        ncf.close()
            
    except Exception as err:
        rpErr = True
        logger.error(err.args[0] + ' for radial file ' + R.file_name)
        return dmRad
            
    #####
    # Update NRT_processed_flag for the processed radial
    #####
        
        if not epErr:
            # Update the local DataFrame
            dmRad['NRT_processed_flag'] = 1      
    
    return
    
def patchRadials(staDF,instacFolder,vers,logger):
    """
    This function patches the radial netCDF files from a single HFR station
    for adding the wmo_platform_code, wigos_id and oceanops_ref global attributes 
    to the files produced with the Euroepan standard data model and for filling 
    the wmo_platform_code global attribute to the files produced with the 
    Copernicus Marine Service In Situ TAC data model.
    
    The WMO code, the OceanOps ref and the WIGOS ID of each radial station are 
    taken from the EU HFR NODE database.
    
    INPUTS:
        staDF: DataFrame containing the information of the stations belonging 
                     to the network to be patched
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
        # Extrct OcenOps ref, WMO code and WIGOS id for the current station
        #####
        
        oceanopsRef = staDF.loc['oceanops_ref']
        wmoCode = staDF.loc['wmo_code']
        wigosId = staDF.loc['wigos_id']
        
        #####
        # List radial nc files to be patched
        #####
        
        # Create the pandas Series used for patching files
        ehnFiles = pd.Series(dtype='object')
        
        # Set input folder path for files built with the European standard data model
        ehnInputFolder = os.path.join(staDF.loc['radial_HFRnetCDF_folder_path'], vers, staDF.loc['station_id'])
        
        # List files built with the European standard data model
        ehnRadList = [file for file in glob.glob(os.path.join(ehnInputFolder,'**/*.nc'), recursive = True)]
        ehnRadList.sort()
        
        # Insert radial nc file paths into the Series used for patching files
        ehnFiles = pd.Series(ehnRadList)
        
        if instacFolder:
            # Create the pandas Series used for patching files
            instacFiles = pd.Series(dtype='object')
            
            # Set input folder path for files built with the Copernicus Marine Service In Situ TAC data model
            instacInputFolder = os.path.join(instacFolder, staDF.loc['network_id'], 'Radials', vers, staDF.loc['station_id'])
            
            # List files built with the Copernicus Marine Service In Situ TAC data model
            instacRadList = [file for file in glob.glob(os.path.join(instacInputFolder,'**/*.nc'), recursive = True)]
            instacRadList.sort()
            
            # Insert radial nc file paths into the Series used for patching files
            instacFiles = pd.Series(instacRadList)
        
        #####
        # Apply the patch
        #####
        
        # Patch radial files built with the European standard data model
        ehnFiles.apply(lambda x: applyEHNradialPatch(x, oceanopsRef, wmoCode, wigosId, logger))
        
        if instacFolder:
            # Patch radial files built with the Copernicus Marine Service In Situ TAC data model
            instacFiles.apply(lambda x: applyINSTACradialPatch(x, wmoCode, logger))
        
        
    except Exception as err:
        pRerr = True
        logger.error(err.args[0])    
    
    return

def patchNetwork(networkID, dataFolder, instacFolder, sqlConfig):
    """
    This function patches the radial netCDF files of a single HFR network
    for adding the wmo_platform_code, wigos_id and oceanops_ref global attributes 
    to the files produced with the Euroepan standard data model and for filling 
    the wmo_platform_code global attribute to the files produced with the 
    Copernicus Marine Service In Situ TAC data model.
    
    The WMO code, the OceanOps ref and the WIGOS ID of each radial station are 
    taken from the EU HFR NODE database.
    
    INPUTS:
        networkID: network ID of the network to be processed
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
        logger = logging.getLogger('EU_HFR_NODE_HIST_wmoCode_ncPatch_v3_' + networkID)
        logger.setLevel(logging.INFO)
        # Create console handler and set level to DEBUG
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # Create logfile handler
        lfh = logging.FileHandler(networkLogFolder + '/EU_HFR_NODE_HIST_wmoCode_ncPatch_v3_' + networkID + '.log')
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
    
    try:
        
    #####
    # Manage data folders
    #####
        
        # Modify data folders (if needed)
        if dataFolder:
            # Modify total data folder paths
            networkData = networkData.apply(lambda x: modifyNetworkDataFolders(x,dataFolder,logger),axis=1)
            # Modify radial data folder paths
            stationData = stationData.apply(lambda x: modifyStationDataFolders(x,dataFolder,logger),axis=1)
            
    #####
    # Apply the patch
    #####
        
        # Apply the patch to radial files
        stationData.apply(lambda x:patchRadials(x,instacFolder,vers,logger),axis=1)
            
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
        opts, args = getopt.getopt(argv,"n:d:i:h",["network=","data-folder=","instac-folder=","help"])
    except getopt.GetoptError:
        print('Usage: wmoCode_ncPatch_v3.py -n <network ID of the network to be processed (if not specified, all the networks are processed)> ' \
              + '-d <full path of the folder containing network data (if not specified, data folder paths read from the database are used)> ' \
                          + '-i <full path of the folder where to save data for Copernicus Marine Service (if not specified, files for Copernicus Marine Service are not patched)>')
        sys.exit(2)
        
    # Initialize optional arguments
    ntw = None
    dataFolder = None
    instacFolder = None
        
    for opt, arg in opts:
        if opt == '-h':
            print('Usage: wmoCode_ncPatch_v3.py -n <network ID of the network to be processed (if not specified, all the networks are processed)> ' \
                  + '-d <full path of the folder containing network data (if not specified, data folder paths read from the database are used)> ' \
                              + '-i <full path of the folder where to save data for Copernicus Marine Service (if not specified, files for Copernicus Marine Service are not patched)>')
            sys.exit()
        elif opt in ("-n", "--network"):
            ntw = arg
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
            
    # Create logger
    logger = logging.getLogger('EU_HFR_NODE_HIST_wmoCode_ncPatch_v3')
    logger.setLevel(logging.INFO)
    # Create console handler and set level to DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # Create logfile handler
    lfh = logging.FileHandler('/var/log/EU_HFR_NODE_HIST/EU_HFR_NODE_HIST_wmoCode_ncPatch_v3.log')
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
    
    logger.info('Historical data patching started.')
    
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
            patchNetwork(ntw, dataFolder, instacFolder, sqlConfig)
        # Process all networks if no specific network is specified for processing
        else:
            # Set the queue containing the network IDs
            networkQueue = networkIDs[:]
            
            # Set the batch size
            numCPU = os.cpu_count()
            if len(networkIDs) < numCPU -1:
                batchDim = len(networkIDs)
            else:
                batchDim = numCPU - 2
                
            # Set the process pool executor for parallel processing
            with futures.ProcessPoolExecutor() as ex:      
                # Set the dictionary of the running processes (contaning processes and related network IDs)
                pool = {}
                # Start a process per each network in the batch
                for ntw in networkIDs[0:batchDim]:
                    # Wait a bit (useful for multiprocessing management)
                    time.sleep(10)
                    # Start the process and insert process and the related network ID into the dictionary of the running processs
                    pool[ex.submit(patchNetwork, ntw, dataFolder, instacFolder, sqlConfig)] = ntw
                    logger.info('Patching for ' + ntw + ' network started')
                    # Pop the network from the queue
                    networkQueue.remove(ntw)
                
                # Check for terminated processes and launch remaining ones
                while pool:
                    # Get the terminated processes
                    done = futures.as_completed(pool)
                    
                    # Relaunch remaining processes
                    for future in done:
                        # Get the ID of the newtork whose process is terminated
                        trmNtw = pool[future]
                        print('Patching for ' + trmNtw + ' network ended')
                        # Pop the process from the dictionary of running processes
                        pool.pop(future)
                        
                        # Check if networks waiting for processing are present in the queue
                        if networkQueue:
                            # Get the next network to be processed from the queue
                            nxtNtw = networkQueue[0]
                            
                            # Wait a bit (useful for multiprocessing management)
                            time.sleep(10)
                            # Start the process and insert process and the related network ID into the dictionary of the running processs
                            pool[ex.submit(patchNetwork, nxtNtw, dataFolder, instacFolder, sqlConfig)] = nxtNtw
                            print('Patching for ' + nxtNtw + ' network started')
                            # Pop the network from the queue
                            networkQueue.remove(nxtNtw)            
    
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
    
    