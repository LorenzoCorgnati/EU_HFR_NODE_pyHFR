#!/usr/bin/python3


# Created on Fri Feb 25 16:16:47 2022

# @author: Lorenzo Corgnati
# e-mail: lorenzo.corgnati@sp.ismar.cnr.it


# This application inserts into the EU HFR NODE EU HFR NODE database the information about 
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
import getopt
import glob
import logging
import datetime as dt
import pandas as pd
import sqlalchemy
from sqlalchemy import text
from dateutil.relativedelta import relativedelta
from radials import Radial, buildEHNradialFolder, buildEHNradialFilename
from totals import Total, buildEHNtotalFolder, buildEHNtotalFilename, combineRadials
from calc import createLonLatGridFromBB, createLonLatGridFromBBwera, createLonLatGridFromTopLeftPointWera
from common import addBoundingBoxMetadata
import pickle
from concurrent import futures
import time

######################
# PROCESSING FUNCTIONS
######################

def applyEHNtotalDataModel(dmTot,networkData,stationData,vers,eng,logger):
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
        eng: SQLAlchemy engine for connecting to the Mysql EU HFR NODE database
        logger: logger object of the current processing

        
    OUTPUTS:
        dmTot = Series containing the processed Total object with the related information
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    dmErr = False
    
    # Check if the Radial was already processed
    if dmTot['NRT_processed_flag'] == 0:
    
        try:        
            # Get the Total object
            T = dmTot['Total']
            
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
            
            # Create netCDF from DataSet and save it
            T.xds.to_netcdf(ncFile, format=T.xds.attrs['netcdf_format'])            
            logger.info(ncFilename + ' total netCDF file succesfully created and stored (' + vers + ').')
            
    #####        
    # Insert information about the created total netCDF into EU HFR NODE database 
    #####
    
            try:
                # Delete the entry with the same filename from total_HFRnetCDF_tb table, if present
                totalDeleteQuery = 'DELETE FROM total_HFRnetCDF_tb WHERE filename=\'' + ncFilename + '\''
                eng.execute(totalDeleteQuery)  
                
                # Prepare data to be inserted into EU HFR NODE database
                if T.is_combined:
                    dataTotalNC = {'filename': [ncFilename], \
                                    'network_id': [networkData.iloc[0]['network_id']], \
                                    'timestamp': [T.time.strftime('%Y %m %d %H %M %S')], 'datetime': [T.time.strftime('%Y-%m-%d %H:%M:%S')], \
                                    'creation_date': [dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")], \
                                    'filesize': [os.path.getsize(ncFile)/1024], 'ttl_filename': T.file_name, 'check_flag': [0]}
                else:                    
                    dataTotalNC = {'filename': [ncFilename], \
                                    'network_id': [networkData.iloc[0]['network_id']], \
                                    'timestamp': [T.time.strftime('%Y %m %d %H %M %S')], 'datetime': [T.time.strftime('%Y-%m-%d %H:%M:%S')], \
                                    'creation_date': [dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")], \
                                    'filesize': [os.path.getsize(ncFile)/1024], 'input_filename': T.file_name, 'check_flag': [0]}
                dfTotalNC = pd.DataFrame(dataTotalNC)
                
                # Insert data into total_HFRnetCDF_tb table
                dfTotalNC.to_sql('total_HFRnetCDF_tb', con=eng, if_exists='append', index=False, index_label=dfTotalNC.columns)
                logger.info(ncFilename + ' total netCDF file information inserted into EU HFR NODE database.')
                
            except sqlalchemy.exc.DBAPIError as err:        
                dMerr = True
                logger.error('MySQL error ' + err._message())        
    
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
            
            # Update the total_input_tb table on the EU HFR NODE database
            if not T.is_combined:
                try:
                    totalUpdateQuery = 'UPDATE total_input_tb SET NRT_processed_flag=1 WHERE filename=\'' + T.file_name + '\''
                    eng.execute(totalUpdateQuery) 
                except sqlalchemy.exc.DBAPIError as err:        
                    dMerr = True
                    logger.error('MySQL error ' + err._message())        
    
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
    
    return  T

def performRadialCombination(combRad,networkData,numActiveStations,vers,eng,logger):
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
        eng: SQLAlchemy engine for connecting to the Mysql EU HFR NODE EU HFR NODE database
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
            if 0 in combRad['NRT_combined_flag'].values:
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
                    for idx in combRad.loc[combRad['extension'] == '.ruv'].loc[:]['Radial'].index:
                        combRad.loc[idx]['Radial'].data.VELO *= 100
                        combRad.loc[idx]['Radial'].data.HCSS *= 10000
                
                # Get the combination search radius in meters
                searchRadius = networkData.iloc[0]['combination_search_radius'] * 1000      # Combination search radius is stored in km in the EU HFR NODE database
                
                # Get the timestamp
                timeStamp = dt.datetime.strptime(str(combRad.iloc[0]['datetime']),'%Y-%m-%d %H:%M:%S')
                
                # Generate the combined Total
                T, warn = combineRadials(combRad,gridGS,searchRadius,gridResolution,timeStamp)
                
                if warn=='':
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
                    
                    # Set the filename (with full path) for the ttl file
                    ttlFilePath = buildEHNtotalFolder(networkData.iloc[0]['total_HFRnetCDF_folder_path'].replace('nc','ttl'),T.time,vers)
                    ttlFilename = buildEHNtotalFilename(networkData.iloc[0]['network_id'],T.time,'.ttl')
                    ttlFile = ttlFilePath + ttlFilename 
                    
                    # Add filename to the Total object
                    T.file_name = ttlFilename
                    
                    # Create the destination folder
                    if not os.path.isdir(ttlFilePath):
                        os.makedirs(ttlFilePath)
                    
                    # Save Total object as .ttl file with pickle
                    with open(ttlFile, 'wb') as ttlFile:
                        pickle.dump(T, ttlFile)
                        
    #####
    # Update NRT_combined_flag for the combined radials                        
    #####
                    # Update the local DataFrame if radials from all station contributed to making the total
                    if len(combRad) == numActiveStations:
                        combRad['NRT_combined_flag'] = combRad['NRT_combined_flag'].replace(0,1)
                    
                        # Update the radial_input_tb table on the EU HFR NODE database
                        try:
                            combRad['Radial'].apply(lambda x: eng.execute('UPDATE radial_input_tb SET NRT_combined_flag=1 WHERE filename=\'' + x.file_name + '\''))
                        except sqlalchemy.exc.DBAPIError as err:        
                            dMerr = True
                            logger.error('MySQL error ' + err._message())
                        
                else:
                    logger.info(warn + ' for network ' + networkData.iloc[0]['network_id'] + ' at timestamp ' + timeStamp.strftime('%Y-%m-%d %H:%M:%S'))
                    return combTot            
        
    except Exception as err:
        crErr = True
        logger.error(err.args[0] + ' for network ' + networkData.iloc[0]['network_id']  + ' in radial combination at timestamp ' + timeStamp.strftime('%Y-%m-%d %H:%M:%S')) 
                 
    return combTot

def applyEHNradialDataModel(dmRad,networkData,radSiteData,vers,eng,logger):
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
        eng: SQLAlchemy engine for connecting to the Mysql EU HFR NODE EU HFR NODE database
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
            
            # Create netCDF from DataSet and save it
            R.xds.to_netcdf(ncFile, format=R.xds.attrs['netcdf_format'])            
            logger.info(ncFilename + ' radial netCDF file succesfully created and stored (' + vers + ').')
            
    #####        
    # Insert information about the created radial netCDF into EU HFR NODE database 
    #####
    
            try:
                # Delete the entry with the same filename from radial_HFRnetCDF_tb table, if present
                radialDeleteQuery = 'DELETE FROM radial_HFRnetCDF_tb WHERE filename=\'' + ncFilename + '\''
                eng.execute(radialDeleteQuery)           
                
                # Prepare data to be inserted into EU HFR NODE database
                dataRadialNC = {'filename': [ncFilename], \
                                'network_id': [radSiteData.iloc[0]['network_id']], 'station_id': [radSiteData.iloc[0]['station_id']], \
                                'timestamp': [R.time.strftime('%Y %m %d %H %M %S')], 'datetime': [R.time.strftime('%Y-%m-%d %H:%M:%S')], \
                                'creation_date': [dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")], \
                                'filesize': [os.path.getsize(ncFile)/1024], 'input_filename': [R.file_name], 'check_flag': [0]}
                dfRadialNC = pd.DataFrame(dataRadialNC)
                
                # Insert data into radial_HFRnetCDF_tb table
                dfRadialNC.to_sql('radial_HFRnetCDF_tb', con=eng, if_exists='append', index=False, index_label=dfRadialNC.columns)
                logger.info(ncFilename + ' radial netCDF file information inserted into EU HFR NODE database.')
                
            except sqlalchemy.exc.DBAPIError as err:        
                dMerr = True
                logger.error('MySQL error ' + err._message())        
    
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
            
            # Update the radial_input_tb table on the EU HFR NODE database
            try:
                radialUpdateQuery = 'UPDATE radial_input_tb SET NRT_processed_flag=1 WHERE filename=\'' + R.file_name + '\''
                eng.execute(radialUpdateQuery) 
            except sqlalchemy.exc.DBAPIError as err:        
                dMerr = True
                logger.error('MySQL error ' + err._message())        
    
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

def updateLastCalibrationDate(lcdRad,radSiteData,eng,logger):
    """
    This function updates, if necessary, the last_calibration_date field of the 
    station_tb table of the EU HFR NODE database based on the input radial file
    metadata.
    
    INPUTS:
        lcdRad: Series containing the radial to be processed with the related information
        radSiteData: DataFrame containing the information of the radial site that produced the radial
        eng: SQLAlchemy engine for connecting to the Mysql EU HFR NODE database
        logger: logger object of the current processing

        
    OUTPUTS:
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    lcdErr = False
    
    try:
        
        # Get the Radial object
        R = lcdRad['Radial']
            
    # Check if the Radial object contains the last pattern date
        if not R.is_wera:
            # Get the last calibration date from Radial object metadata
            lcdFromFile = dt.datetime.strptime(R.metadata['PatternDate'],'%Y %m %d %H %M %S').date()
            # Check if the last calibration date is to be updated
            if lcdFromFile > radSiteData.iloc[0]['last_calibration_date']:
                # Update the station_tb table on the EU HFR NODE database
                try:
                    stationUpdateQuery = 'UPDATE station_tb SET last_calibration_date=\'' + lcdFromFile.strftime('%Y-%m-%d') + '\' WHERE station_id=\'' + radSiteData.iloc[0]['station_id'] + '\''
                    eng.execute(stationUpdateQuery)
                    logger.info('Last calibration date updated for station ' + radSiteData.iloc[0]['station_id'])
                except sqlalchemy.exc.DBAPIError as err:        
                    lcdErr = True
                    logger.error('MySQL error ' + err._message())   
                
    except Exception as err:
        lcdErr = True
        logger.error(err.args[0] + ' for radial file ' + R.file_name)
        return
    
    return

def processTotals(dfTot,networkID,networkData,stationData,startDate,eng,logger):
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
    pTerr = False
    
    try:
        logger.info('Total processing started for ' + networkID + ' network ' + '(' + vers + ').') 
        
        #####
        # Enhance the total DataFrame
        #####
        
        # Add Total objects to the DataFrame
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
        
        dfTot = dfTot.apply(lambda x: applyEHNtotalDataModel(x,networkData,stationData,vers,eng,logger),axis=1)
        
    except Exception as err:
        pTerr = True
        logger.error(err.args[0])    
    
    return
    
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
        
        #####        
        # Update the last calibration date in station_tb table of teh database
        #####
        
        groupedRad.apply(lambda x: updateLastCalibrationDate(x,stationData.loc[stationData['station_id'] == x.station_id],eng,logger),axis=1)
        
        #####        
        # Apply QC to Radials
        #####
        
        groupedRad['Radial'] = groupedRad.apply(lambda x: applyEHNradialQC(x,stationData.loc[stationData['station_id'] == x.station_id],vers,logger),axis=1)
        
        #####        
        # Convert Radials to standard data format (netCDF)
        #####
        
        groupedRad = groupedRad.apply(lambda x: applyEHNradialDataModel(x,networkData,stationData.loc[stationData['station_id'] == x.station_id],vers,eng,logger),axis=1)
                
        #####
        # Combine Radials into Total
        #####
        
        dfTot = performRadialCombination(groupedRad,networkData,numActiveStations,vers,eng,logger)
        
        if dfTot.size > 0:
        
        #####        
        # Apply QC to Totals
        #####
        
            dfTot['Total'] = dfTot.apply(lambda x: applyEHNtotalQC(x,networkData,vers,logger),axis=1)        
            
        #####        
        # Convert Total to standard data format (netCDF)
        #####
        
            dfTot = dfTot.apply(lambda x: applyEHNtotalDataModel(x,networkData,stationData,vers,eng,logger),axis=1)
        
    except Exception as err:
        pRerr = True
        logger.error(err.args[0])    
    
    return

def selectTotals(networkID,startDate,eng,logger):
    """
    This function selects the totals to be processed for the input network by reading
    from the total_input_tb table of the EU HFR NODE database.
        
    INPUTS:
        networkID: network ID of the network to be processed
        startDate: string containing the datetime of the starting date of the processing period
        eng: SQLAlchemy engine for connecting to the Mysql EU HFR NODE database
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
    
    try:
       
        #####
        # Select totals to be processed
        #####
        
        # Set and execute the query for getting totals to be processed
        if 'HFR-US' in networkID:
            # GESTIRE TOTALI RETI US
            
            print('TO BE DONE')
        else:
            networkStr = '\'' + networkID + '\''
        try:
            totalSelectionQuery = 'SELECT * FROM total_input_tb WHERE datetime>=\'' + startDate + '\' AND (network_id=' + networkStr + ') AND (NRT_processed_flag=0) ORDER BY TIMESTAMP'
            totalsToBeProcessed = pd.read_sql(totalSelectionQuery, con=eng)
        except sqlalchemy.exc.DBAPIError as err:        
            sTerr = True
            logger.error('MySQL error ' + err._message())
                
    except Exception as err:
        sTerr = True
        logger.error(err.args[0] + ' for network ' + networkID)
            
    return totalsToBeProcessed

def selectRadials(networkID,startDate,eng,logger):
    """
    This function selects the radials to be processed for the input network by reading
    from the radial_input_tb table of the EU HFR NODE EU HFR NODE database.
        
    INPUTS:
        networkID: network ID of the network to be processed
        startDate: string containing the datetime of the starting date of the processing period
        eng: SQLAlchemy engine for connecting to the Mysql EU HFR NODE EU HFR NODE database
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
            radialSelectionQuery = 'SELECT * FROM radial_input_tb WHERE datetime>=\'' + startDate + '\' AND (network_id=' + networkStr + ') AND (NRT_processed_flag=0 OR NRT_combined_flag=0) ORDER BY TIMESTAMP'
            radialsToBeProcessed = pd.read_sql(radialSelectionQuery, con=eng)
        except sqlalchemy.exc.DBAPIError as err:        
            sRerr = True
            logger.error('MySQL error ' + err._message())
                
    except Exception as err:
        sRerr = True
        logger.error(err.args[0] + ' for network ' + networkID)
            
    return radialsToBeProcessed


def inputTotals(networkID,networkData,startDate,eng,logger):
    """
    This function lists the input total files pushed by the HFR data providers 
    and inserts into the EU HFR NODE EU HFR NODE database the information needed for the 
    generation of the total data files into the European standard data model.
    
    INPUTS:
        networkID: network ID of the network to be processed
        networkData: DataFrame containing the information of the network to be processed
        startDate: string containing the datetime of the starting date of the processing period
        eng: SQLAlchemy engine for connecting to the Mysql EU HFR NODE EU HFR NODE database
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
                        # Check if the file is already present in the EU HFR NODE database
                        try:
                            totalPresenceQuery = 'SELECT * FROM total_input_tb WHERE datetime>\'' + startDate + '\' AND filename=\'' + fileName + '\''
                            totalPresenceData = pd.read_sql(totalPresenceQuery, con=eng)
                            numPresentTotals = totalPresenceData.shape[0]
                            if numPresentTotals==0:    # the current file is not present in the EU HFR NODE database
                                # Get file timestamp
                                total = Total(inputFile)
                                timeStamp = total.time.strftime("%Y %m %d %H %M %S")                    
                                dateTime = total.time.strftime("%Y-%m-%d %H:%M:%S")  
                                # Get file size in Kbytes
                                fileSize = os.path.getsize(inputFile)/1024    
    #####
    # Insert total information into EU HFR NODE database
    #####
                                # Prepare data to be inserted into EU HFR NODE database
                                dataTotal = {'filename': [fileName], 'filepath': [filePath], 'network_id': [networkID], 'timestamp': [timeStamp], \
                                             'datetime': [dateTime], 'reception_date': [dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")], \
                                              'filesize': [fileSize], 'extension': [fileExt], 'NRT_processed_flag': [0]}
                                dfTotal = pd.DataFrame(dataTotal)
                                
                                # Insert data into EU HFR NODE database
                                dfTotal.to_sql('total_input_tb', con=eng, if_exists='append', index=False, index_label=dfTotal.columns)
                                logger.info(fileName + ' total file information inserted into EU HFR NODE database.')  
                        except sqlalchemy.exc.DBAPIError as err:        
                            iTerr = True
                            logger.error('MySQL error ' + err._message())
                    except Exception as err:
                        iTerr = True
                        logger.error(err.args[0] + ' for file ' + fileName)
                    
    except Exception as err:
        iTerr = True
        logger.error(err.args[0])
    
    return


def inputRadials(networkID,stationData,startDate,eng,logger):
    """
    This function lists the input radial files pushed by the HFR data providers 
    and inserts into the EU HFR NODE EU HFR NODE database the information needed for the 
    combination of radial files into totals and for the generation of the 
    radial and total data files into the European standard data model.
    
    INPUTS:
        networkID: network ID of the network to be processed
        stationData: DataFrame containing the information of the stations belonging 
                     to the network to be processed
        startDate: string containing the datetime of the starting date of the processing period
        eng: SQLAlchemy engine for connecting to the Mysql EU HFR NODE EU HFR NODE database
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
                            # Check if the file is already present in the EU HFR NODE database
                            try:
                                radialPresenceQuery = 'SELECT * FROM radial_input_tb WHERE datetime>\'' + startDate + '\' AND filename=\'' + fileName + '\''
                                radialPresenceData = pd.read_sql(radialPresenceQuery, con=eng)
                                numPresentRadials = radialPresenceData.shape[0]
                                if numPresentRadials==0:    # the current file is not present in the EU HFR NODE database
                                    # Get file timestamp
                                    radial = Radial(inputFile)
                                    timeStamp = radial.time.strftime("%Y %m %d %H %M %S")                    
                                    dateTime = radial.time.strftime("%Y-%m-%d %H:%M:%S")  
                                    # Get file size in Kbytes
                                    fileSize = os.path.getsize(inputFile)/1024 
    #####
    # Insert radial information into EU HFR NODE database
    #####
                                    # Prepare data to be inserted into EU HFR NODE database
                                    dataRadial = {'filename': [fileName], 'filepath': [filePath], 'network_id': [networkID], \
                                                  'station_id': [stationID], 'timestamp': [timeStamp], 'datetime': [dateTime], \
                                                  'reception_date': [dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")], \
                                                  'filesize': [fileSize], 'extension': [fileExt], 'NRT_processed_flag': [0], \
                                                  'NRT_processed_flag_integrated_network': [0], 'NRT_combined_flag': [0]}
                                    dfRadial = pd.DataFrame(dataRadial)
                                    
                                    # Insert data into EU HFR NODE database
                                    dfRadial.to_sql('radial_input_tb', con=eng, if_exists='append', index=False, index_label=dfRadial.columns)
                                    logger.info(fileName + ' radial file information inserted into EU HFR NODE database.')   
                            except sqlalchemy.exc.DBAPIError as err:        
                                iRerr = True
                                logger.error('MySQL error ' + err._message())
                        except Exception as err:
                            iRerr = True
                            logger.error(err.args[0] + ' for file ' + fileName)
                        
        except Exception as err:
            iRerr = True
            logger.error(err.args[0] + ' for station ' + stationID)
    
    return


def processNetwork(networkID,memory,sqlConfig):
    """
    This function processes the radial and total files of a single HFR network
    for generating radial and total files according to the European standard data model.
    
    The first processing step consists in the listing of the input files
    (both radial and total) pushed by the HFR data providers for inserting into
    the EU HFR NODE EU HFR NODE database the information needed for the combination of radial files
    into totals and for the generation of the radial and total data files according
    to the European standard data model.
    
    The second processing step consists in reading the EU HFR NODE EU HFR NODE database for collecting
    information about the radial data files to be combined into totals and in
    combining and generating radial and total data files according to the European
    standard data model.
    
    The third processing step consists in reading the EU HFR NODE EU HFR NODE database for collecting
    information about the total data files to be converted into the European standard
    data model and in the generating total data files according to the European
    standard data model.
    
    INPUTS:
        networkID: network ID of the network to be processed
        memory: number of days in the past when to start processing
        sqlConfig: parameters for connecting to the Mysql EU HFR NODE EU HFR NODE database

        
    OUTPUTS:
        pNerr: error flag (True = errors occurred, False = no error occurred)
        
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
        startDate = (dt.datetime.utcnow()- relativedelta(days=memory)).strftime("%Y-%m-%d")
        
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
        logger.info(networkID + ' station data successfully fetched from EU HFR NODE database.')
    except sqlalchemy.exc.DBAPIError as err:        
        pNerr = True
        logger.error('MySQL error ' + err._message())
        logger.info('Exited with errors.')
        return pNerr
        
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
        logger.info('Radials to be processed successfully selected for network ' + networkID)
        
        # Process radials
        radialsToBeProcessed.groupby('datetime').apply(lambda x:processRadials(x,networkID,networkData,stationData,startDate,eng,logger))
        
        if networkData.iloc[0]['radial_combination'] == 0:
            # Select totals to be processed
            totalsToBeProcessed = selectTotals(networkID,startDate,eng,logger)
            logger.info('Totals to be processed successfully selected for network ' + networkID)
            
            # Process totals
            totalsToBeProcessed.groupby('datetime').apply(lambda x:processTotals(x,networkID,networkData,stationData,startDate,eng,logger))
            
        # Wait a bit (useful for multiprocessing management)
        time.sleep(60)
            
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
    # Set parameter for Mysql EU HFR NODE database connection
    sqlConfig = {
      'user': 'HFRuserCP',
      'password': '!_kRIVAHYH2RLpmQxz_!',
      'host': '150.145.136.108',
      'EU HFR NODE database': 'HFR_node_db',
    }
    
    # Initialize error flag
    EHNerr = False
    
    logger.info('Processing started.')
    
#####
# Network data collection
#####
    
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
        # Create multiprocessing pool
        with futures.ProcessPoolExecutor() as ex:
            # Launch processes for each network
            pool = {}
            for ntw in networkIDs:
                pool[ex.submit(processNetwork, ntw, memory, sqlConfig)] = ntw
                logger.info('Job for processing ' + ntw + ' submitted')
            
            while pool:
                # Check for status of the futures which are currently working
                done, not_done = futures.wait(pool, return_when=futures.FIRST_COMPLETED)
                
                # Resubmit terminated processes
                for future in done:
                    ntw = pool[future]
                    if future.result():
                        logger.error('Job for processing ' + ntw + ' network exited with errors')
                    pool.pop(future)
                    pool[ex.submit(processNetwork, ntw, memory, sqlConfig)] = ntw
                    logger.info('Job for processing ' + ntw + ' network resubmitted')
    
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
    
    