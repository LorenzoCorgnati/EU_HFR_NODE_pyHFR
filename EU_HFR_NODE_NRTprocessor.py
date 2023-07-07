#!/usr/bin/python3


# Created on Fri Feb 25 16:16:47 2022

# @author: Lorenzo Corgnati
# e-mail: lorenzo.corgnati@sp.ismar.cnr.it


# This application inserts into the EU HFR NODE database the information about 
# radial and total HFR files (both Codar and WERA) pushed by the data providers,
# combines radials into totals and generates HFR radial and total data to netCDF 
# files according to the European standard data model for data distribution towards
# Copernicus Marine Service In Situ component.

# When calling the application it is possible to specify the number of days in 
# the past when to start processing (default to 3).

# This application implements parallel computing by launching a separate process 
# per each HFR network to be processed.

import os
import sys
import io
import getopt
import glob
import logging
import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import mysql.connector as sql
from mysql.connector import errorcode
import sqlalchemy
from dateutil.relativedelta import relativedelta
from radials import Radial, buildEHNradialFolder, buildEHNradialFilename
from totals import Total, buildEHNtotalFolder, buildEHNtotalFilename
from calc import true2mathAngle, createLonLatGridFromBB, createLonLatGridFromBBwera, createLonLatGridFromTopLeftPointWera
from pyproj import Geod
import latlon
import time
import math
import pickle

######################
# PROCESSING FUNCTIONS
######################

def applyEHNradialDataModel(dmRad,networkData,radSiteData,vers,logger):
    """
    This function applies the European standard data model to radial object and saves
    the resulting netCDF file. The Radial object is also saved as .rdl file via pickle
    binary serialization.
    
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
    
    try:
    
    #####
    # Enhance Radial object with information from database
    #####
    
        # Get the radial object
        R = dmRad['Radial']
        
        # Add metadata related to range limits
        R.metadata['RangeMin'] = '0 km'
        if 'RangeResolutionKMeters' in R.metadata:
            R.metadata['RangeMax'] = str(float(R.metadata['RangeResolutionKMeters'].split()[0])*(radSiteData.iloc[0]['number_of_range_cells']-1)) + ' km'
        elif 'RangeResolutionMeters' in R.metadata:
            R.metadata['RangeMax'] = str((float(R.metadata['RangeResolutionMeters'].split()[0]) * 0.001)*(radSiteData.iloc[0]['number_of_range_cells']-1)) + ' km'
        
    #####        
    # Convert to standard data format (netCDF)  
    #####
    
        # Check if the Radial was already processed
        if dmRad['NRT_processed_flag'] == 0:
            # Apply the standard data model
            R.apply_ehn_datamodel(networkData,radSiteData,vers)
    
    
    
    #####
    # Save Radial object as .rdl file with pickle
    #####
    
        # with open('filename.rdl', 'wb') as rdlFile:
        #      pickle.dump(R, rdlFile)
    
    
        
    except Exception as err:
        dmErr = True
        logger.error(err.args[0] + ' for radial file ' + R.file_name)     
    
    return  dmRad

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
                        R = pickle.load(rdlFile)
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
        
    except Exception as err:
        qcErr = True
        logger.error(err.args[0] + ' for radial file ' + R.file_name)     
    
    return  R    
    
def processRadials(groupedRad,networkID,networkData,stationData,startDate,eng,logger):
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
        startDate: string containing the datetime of the starting date of the processing period
        eng: SQLAlchemy engine for connecting to the Mysql EU HFR NODE database
        logger: logger object of the current processing

        
    OUTPUTS:
        
    """
    #####
    # Setup
    #####
    
    # Set the version of the data model
    vers = 'v3'
    
    # Initialize error flag
    pRerr = False
    
    # Retrieve the number of operational stations
    numActiveStations = stationData['operational_to'].isna().sum() 
    
    try:
        logger.info('Radial processing started for ' + networkID + ' network ' + '(' + vers + ').') 

        #####
        # Enhance the radial DataFrame
        #####
        
        # Add Radial objects to the DataFrame
        groupedRad['Radial'] = (groupedRad.filepath + '/' + groupedRad.filename).apply(lambda x: Radial(x))
        
        # Rename indices with site codes
        indexMapper = dict(zip(groupedRad.index.values.tolist(),groupedRad['station_id'].to_list()))
        groupedRad.rename(index=indexMapper,inplace=True)        
        
        #####        
        # Apply radial data QC    
        #####
        
        groupedRad['Radial'] = groupedRad.apply(lambda x: applyEHNradialQC(x,stationData.loc[stationData['station_id'] == x.station_id],vers,logger),axis=1)
        
        #####        
        # Radial data conversion to standard data format (netCDF)
        #####
        
        groupedRad = groupedRad.apply(lambda x: applyEHNradialDataModel(x,networkData,stationData.loc[stationData['station_id'] == x.station_id],vers,logger),axis=1)
                
        
        
        
        
        #####        
        # Insert radial information into database    
        #####
        
        # TO BE DONE
        
        #####
        # Radial combination into totals
        #####
        
        # TO BE DONE - da fare solo per radiali con NRT_combined_flag=0
        # # INSERIRE lonMin, lonMax, latMin, latMax, gridResolution IN T.metadata - CHECK NOMI METADATI
        # T.metadata['BBminLongitude'] = str(lonMin) + ' deg'
        # T.metadata['BBmaxLongitude'] = str(lonMax) + ' deg'
        # T.metadata['BBminLatitude'] = str(latMin) + ' deg'
        # T.metadata['BBmaxLatitude'] = str(latMax) + ' deg'
        # T.metadata['GridSpacing'] = str(gridResolution/1000) + ' km'
        # # INSERIRE ATTRIBUTO is_wera
        # if weraGrid:
        #     T.is_wera = True
        # else:
        #     T.is_wera = False
        
        # # Total data QC
        
        # # Save Total object as .ttl file with pickle
        # with open('filename.ttl', 'wb') as ttlFile:
        #     pickle.dump(T, ttlFile)
            
        #####
        # Total data conversion to standard format (netCDF)
        #####
        
        # TO BE DONE
        
        #####        
        # Insert information into database    
        #####
        
        # TO BE DONE
        
    except Exception as err:
        pRerr = True
        logger.error(err.strerror)    
    
    return

def selectRadials(networkID,startDate,eng,logger):
    """
    This function selects the radials to be processed for the input network by reading
    from the radial_input_tb table of the EU HFR NODE database.
        
    INPUTS:
        networkID: network ID of the network to be processed
        startDate: string containing the datetime of the starting date of the processing period
        eng: SQLAlchemy engine for connecting to the Mysql EU HFR NODE database
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
    
    try:
       
        #####
        # Select radials to be processed
        #####
        
        # Set and execute the query for getting radials to be processed
        if networkID == 'HFR-WesternItaly':
            networkStr = '\'HFR-TirLig\' OR network_id=\'HFR-LaMMA\' OR network_id=\'HFR-ARPAS\''
        else:
            networkStr = '\'' + networkID + '\''
        try:
            radialSelectionQuery = 'SELECT * FROM radial_input_tb WHERE datetime>\'' + startDate + '\' AND (network_id=' + networkStr + ') AND (NRT_processed_flag=0 OR NRT_combined_flag=0) ORDER BY TIMESTAMP'
            radialsToBeProcessed = pd.read_sql(radialSelectionQuery, con=eng)
        except sqlalchemy.exc.DBAPIError as err:        
            sRerr = True
            logger.error('MySQL error ' + err._message())
                
    except Exception as err:
        sRerr = True
        logger.error(err.strerror + ' for network ' + networkID)
            
    return radialsToBeProcessed


def inputTotals(networkID,networkData,startDate,eng,logger):
    """
    This function lists the input total files pushed by the HFR data providers 
    and inserts into the EU HFR NODE database the information needed for the 
    generation of the total data files into the European standard data model.
    
    INPUTS:
        networkID: network ID of the network to be processed
        networkData: DataFrame containing the information of the network to be processed
        startDate: string containing the datetime of the starting date of the processing period
        eng: SQLAlchemy engine for connecting to the Mysql EU HFR NODE database
        logger: logger object of the current processing

        
    OUTPUTS:
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    iTerr = False
    
    # Convert starting date from string to timestamp
    mTime = dt.datetime.strptime(startDate,"%Y-%m-%d").timestamp()
    
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
                # List input files (only in the processing period)
                codarInputFiles = [file for file in glob.glob(os.path.join(inputFolder,codarTypeWildcard), recursive = True) if os.path.getmtime(file) >= mTime]                    
                weraInputFiles = [file for file in glob.glob(os.path.join(inputFolder,weraTypeWildcard), recursive = True) if os.path.getmtime(file) >= mTime]
                inputFiles = codarInputFiles + weraInputFiles
                for inputFile in inputFiles:
                    try:
                        # Get file parts
                        filePath = os.path.dirname(inputFile)
                        fileName = os.path.basename(inputFile)
                        fileExt = os.path.splitext(inputFile)[1]
                        # Check if the file is already present in the database
                        try:
                            totalPresenceQuery = 'SELECT * FROM total_input_tb WHERE datetime>\'' + startDate + '\' AND filename=\'' + fileName + '\''
                            totalPresenceData = pd.read_sql(totalPresenceQuery, con=eng)
                            numPresentTotals = totalPresenceData.shape[0]
                            if numPresentTotals==0:    # the current file is not present in the database
                                # Get file timestamp
                                total = Total(inputFile)
                                timeStamp = total.time.strftime("%Y %m %d %H %M %S")                    
                                dateTime = total.time.strftime("%Y-%m-%d %H:%M:%S")  
                                # Get file size in Kbytes
                                fileSize = os.path.getsize(inputFile)/1024    
    #####
    # Insert total information into database
    #####
                                # Prepare data to be inserted into database
                                dataTotal = {'filename': [fileName], 'filepath': [filePath], 'network_id': [networkID], 'timestamp': [timeStamp], \
                                             'datetime': [dateTime], 'reception_date': [dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")], \
                                              'filesize': [fileSize], 'extension': [fileExt], 'NRT_processed_flag': [0]}
                                dfTotal = pd.DataFrame(dataTotal)
                                
                                # Insert data into database
                                dfTotal.to_sql('total_input_tb', con=eng, if_exists='append', index=False, index_label=dfTotal.columns)
                                logger.info(fileName + ' total file information inserted into database.')  
                        except sqlalchemy.exc.DBAPIError as err:        
                            iTerr = True
                            logger.error('MySQL error ' + err._message())
                    except Exception as err:
                        iTerr = True
                        logger.error(err.strerror + ' for file ' + fileName)
                    
    except Exception as err:
        iTerr = True
        logger.error(err.strerror)
    
    return


def inputRadials(networkID,stationData,startDate,eng,logger):
    """
    This function lists the input radial files pushed by the HFR data providers 
    and inserts into the EU HFR NODE database the information needed for the 
    combination of radial files into totals and for the generation of the 
    radial and total data files into the European standard data model.
    
    INPUTS:
        networkID: network ID of the network to be processed
        stationData: DataFrame containing the information of the stations belonging 
                     to the network to be processed
        startDate: string containing the datetime of the starting date of the processing period
        eng: SQLAlchemy engine for connecting to the Mysql EU HFR NODE database
        logger: logger object of the current processing

        
    OUTPUTS:
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    iRerr = False
    
    # Convert starting date from string to timestamp
    mTime = dt.datetime.strptime(startDate,"%Y-%m-%d").timestamp()
    
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
                    # List input files (only in the processing period)
                    inputFiles = [file for file in glob.glob(os.path.join(inputFolder,fileTypeWildcard), recursive = True) if os.path.getmtime(file) >= mTime]                    
                    for inputFile in inputFiles:
                        try:
                            # Get file parts
                            filePath = os.path.dirname(inputFile)
                            fileName = os.path.basename(inputFile)
                            fileExt = os.path.splitext(inputFile)[1]
                            # Check if the file is already present in the database
                            try:
                                radialPresenceQuery = 'SELECT * FROM radial_input_tb WHERE datetime>\'' + startDate + '\' AND filename=\'' + fileName + '\''
                                radialPresenceData = pd.read_sql(radialPresenceQuery, con=eng)
                                numPresentRadials = radialPresenceData.shape[0]
                                if numPresentRadials==0:    # the current file is not present in the database
                                    # Get file timestamp
                                    radial = Radial(inputFile)
                                    timeStamp = radial.time.strftime("%Y %m %d %H %M %S")                    
                                    dateTime = radial.time.strftime("%Y-%m-%d %H:%M:%S")  
                                    # Get file size in Kbytes
                                    fileSize = os.path.getsize(inputFile)/1024    
    #####
    # Insert radial information into database
    #####
                                    # Prepare data to be inserted into database
                                    dataRadial = {'filename': [fileName], 'filepath': [filePath], 'network_id': [networkID], \
                                                  'station_id': [stationID], 'timestamp': [timeStamp], 'datetime': [dateTime], \
                                                  'reception_date': [dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")], \
                                                  'filesize': [fileSize], 'extension': [fileExt], 'NRT_processed_flag': [0], \
                                                  'NRT_processed_flag_integrated_network': [0], 'NRT_combined_flag': [0]}
                                    dfRadial = pd.DataFrame(dataRadial)
                                    
                                    # Insert data into database
                                    dfRadial.to_sql('radial_input_tb', con=eng, if_exists='append', index=False, index_label=dfRadial.columns)
                                    logger.info(fileName + ' radial file information inserted into database.')   
                            except sqlalchemy.exc.DBAPIError as err:        
                                iRerr = True
                                logger.error('MySQL error ' + err._message())
                        except Exception as err:
                            iRerr = True
                            logger.error(err.strerror + ' for file ' + fileName)
                        
        except Exception as err:
            iRerr = True
            logger.error(err.strerror + ' for station ' + stationID)
    
    return


def processNetwork(networkID,memory,sqlConfig):
    """
    This function processes the radial and total files of a single HFR network
    for generating radial and total files according to the European standard data model.
    
    The first processing step consists in the listing of the input files
    (both radial and total) pushed by the HFR data providers for inserting into
    the EU HFR NODE database the information needed for the combination of radial files
    into totals and for the generation of the radial and total data files according
    to the European standard data model.
    
    The second processing step consists in reading the EU HFR NODE database for collecting
    information about the radial data files to be combined into totals and in
    combining and generating radial and total data files according to the European
    standard data model.
    
    The third processing step consists in reading the EU HFR NODE database for collecting
    information about the total data files to be converted into the European standard
    data model and in the generating total data files according to the European
    standard data model.
    
    INPUTS:
        networkID: network ID of the network to be processed
        memory: number of days in the past when to start processing
        sqlConfig: parameters for connecting to the Mysql EU HFR NODE database

        
    OUTPUTS:
        
    """
    #####
    # Setup
    #####
    
    try:
        # Create the folder for the network log
        networkLogFolder = '/var/log/EU_HFR_NODE_NRT/' + networkID
        if not os.path.isdir(networkLogFolder):
            os.mkdir(networkLogFolder)
               
        # Create logger
        logger = logging.getLogger('EU_HFR_NODE_NRT_' + networkID)
        logger.setLevel(logging.INFO)
        # Create console handler and set level to DEBUG
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # Create logfile handler
        lfh = logging.FileHandler(networkLogFolder + '/EU_HFR_NODE_NRT_' + networkID + '.log')
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
        
        # Set datetime of the starting date of the processing period
        startDate = (dt.datetime.now()- relativedelta(days=memory)).strftime("%Y-%m-%d")
        
    except Exception as err:
        pNerr = True
        logger.error(err.strerror)
        logger.info('Exited with errors.')
        return
    
    #####
    # Retrieve information from database
    #####
    
    try:
        # Create SQLAlchemy engine for connecting to database
        eng = sqlalchemy.create_engine('mysql+mysqlconnector://' + sqlConfig['user'] + ':' + \
                                       sqlConfig['password'] + '@' + sqlConfig['host'] + '/' + \
                                       sqlConfig['database'])
        
        # Set and execute the query and get the HFR network data
        networkSelectQuery = 'SELECT * FROM network_tb WHERE network_id=\'' + networkID + '\''
        networkData = pd.read_sql(networkSelectQuery, con=eng)
        numNetworks = networkData.shape[0]
        logger.info(networkID + ' network data successfully fetched from database.')
        # Set and execute the query and get the HFR station data
        if networkID == 'HFR-WesternItaly':
            stationSelectQuery = 'SELECT * FROM station_tb WHERE network_id=\'HFR-TirLig\' OR network_id=\'HFR-LaMMA\' OR network_id=\'HFR-ARPAS\''
        else:
            stationSelectQuery = 'SELECT * FROM station_tb WHERE network_id=\'' + networkID + '\''
        stationData = pd.read_sql(stationSelectQuery, con=eng)
        numStations = stationData.shape[0]
        logger.info(networkID + ' station data successfully fetched from database.')
    except sqlalchemy.exc.DBAPIError as err:        
        pNerr = True
        logger.error('MySQL error ' + err._message())
        logger.info('Exited with errors.')
        return
        
    #####
    # Input HFR data
    #####
    
    try:
        # Input radial data
        inputRadials(networkID, stationData, startDate, eng, logger)
        # Input total data
        inputTotals(networkID, networkData, startDate, eng, logger)
        
        #####
        # Process HFR data
        #####
        
        # Select radials to be processed
        radialsToBeProcessed = selectRadials(networkID,startDate,eng,logger)
        
        # Process radials
        radialsToBeProcessed.groupby('datetime').apply(lambda x:processRadials(x,networkID,networkData,stationData,startDate,eng,logger))
            
            
        
        
        # Selection of totals to be converted based on timestamp
        
        # Total data QC
        
        # INSERIRE lonMin, lonMax, latMin, latMax, gridResolution IN T.metadata - CHECK NOMI METADATI
        T.metadata['BBminLongitude'] = str(lonMin) + ' deg'
        T.metadata['BBmaxLongitude'] = str(lonMax) + ' deg'
        T.metadata['BBminLatitude'] = str(latMin) + ' deg'
        T.metadata['BBmaxLatitude'] = str(latMax) + ' deg'
        
        # Save Total object as .ttl file with pickle
        with open('filename.ttl', 'wb') as ttlFile:
            pickle.dump(T, ttlFile)
        
        # Total data conversion to standard format (netCDF)
        
    except Exception as err:
        pNerr = True
        logger.error(err.strerror)
        logger.info('Exited with errors.')
        return
    
    
    return

####################
# MAIN DEFINITION
####################

def main(argv):
    
#####
# Setup
#####
       
    # Set the argument structure
    try:
        opts, args = getopt.getopt(argv,"m:h",["memory=","help"])
    except getopt.GetoptError:
        print('Usage: EU_HFR_NODE_NRTprocessor.py -m <number of days in the past when to start processing (default to 3)>')
        sys.exit(2)
        
    if not argv:
        memory =3       # number of days in the past when to start processing (default to 3)
        
    for opt, arg in opts:
        if opt == '-h':
            print('EU_HFR_NODE_NRTprocessor.py -m <number of days in the past when to start processing (default to 3)>')
            sys.exit()
        elif opt in ("-m", "--memory"):
            memory = int(arg)
          
    # Create logger
    logger = logging.getLogger('EU_HFR_NODE_NRT')
    logger.setLevel(logging.INFO)
    # Create console handler and set level to DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # Create logfile handler
    lfh = logging.FileHandler('/var/log/EU_HFR_NODE_NRT/EU_HFR_NODE_NRT.log')
    lfh.setLevel(logging.INFO)
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] -- %(levelname)s -- %(module)s - %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    # Add formatter to lfh and ch
    lfh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # Add lfh and ch to logger
    logger.addHandler(lfh)
    logger.addHandler(ch)
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!TO BE CHANGED FOR OPERATIONS (IP set to 150.145.136.104) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Set parameter for Mysql database connection
    sqlConfig = {
      'user': 'HFRuserCP',
      'password': '!_kRIVAHYH2RLpmQxz_!',
      'host': '150.145.136.108',
      'database': 'HFR_node_db',
    }
    
    # Initialize error flag
    EHNerr = False
    
    logger.info('Processing started.')
    
#####
# Network data collection
#####
    
    try:
        # Create SQLAlchemy engine for connecting to database
        eng = sqlalchemy.create_engine('mysql+mysqlconnector://' + sqlConfig['user'] + ':' + \
                                       sqlConfig['password'] + '@' + sqlConfig['host'] + '/' + \
                                       sqlConfig['database'])
            
        # Set and execute the query and get the HFR network IDs to be processed
        networkSelectQuery = 'SELECT network_id FROM network_tb WHERE EU_HFR_processing_flag=1'
        networkIDs = pd.read_sql(networkSelectQuery, con=eng)
        numNetworks = networkIDs.shape[0]
        logger.info('Network IDs successfully fetched from database.')
    except sqlalchemy.exc.DBAPIError as err:        
        EHNerr = True
        logger.error('MySQL error ' + err._message())
        logger.info('Exited with errors.')
        sys.exit()
    
#####
# Process launch and monitor
#####

# TO BE DONE USING MULTIPROCESSING - AAGIUNGERE try except
    try:
        i = 0
        processNetwork(networkIDs.iloc[i]['network_id'], memory, sqlConfig)
        # INSERIRE LOG CHE INDICA INIZIO PROCESSING PER OGNI RETE QUANDO VIENE LANCIATO IL PROCESSO
    
    except Exception as err:
        EHNerr = True
        logger.error(err.strerror)
    
    
####################
    
    if(not EHNerr):
        logger.info('Successfully executed.')
    else:
        logger.info('Exited with errors.')
            
####################


#####################################
# SCRIPT LAUNCHER
#####################################    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    