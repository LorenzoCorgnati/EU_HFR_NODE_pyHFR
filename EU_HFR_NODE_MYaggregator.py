#!/usr/bin/python3


# Created on Thu Mar 7 10:01:25 2024

# @author: Lorenzo Corgnati
# e-mail: lorenzo.corgnati@sp.ismar.cnr.it


# This application creates the temporally aggregated netCDF files for HFR MY prodcuts
# to be delivered to Copernicus Marine Service In Situ TAC.

# The application aggregates the daily aggregated netCDF files produced for HFR NRT
# products, i.e. it works on data files already generated for the NRT workflow.
# In case the MY product has to be created for files not yet processed for the NRT
# workflow, the script EU_HFR_NODE_HISTprocessor.py has to be run before this, in 
# order to create the daily aggregated files according to the Copernicus Marine Service 
# data model.

# THe application reads from the EU HFR NODE EU HFR NODE database the information about 
# radial and total HFR files (both Codar and WERA) pushed by the data providers.

# When calling the application it is possible to specify if all the networks have to be processed
# or only the selected one, the time interval for aggregation and if the compression for 
# netCDF files has to be applied.

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
import sqlalchemy
from dateutil.relativedelta import relativedelta
from concurrent import futures
import time
import xarray as xr
import netCDF4 as nc4
import json

######################
# PROCESSING FUNCTIONS
######################

def adjustToMYINSTACtotalDatamodel(tDS, networkData):
    """
    This function adjusts the data model of the input aggregated radial dataset for complying with
    the Copernicus Marine Service data model for MY products.
    the input dataset must follow the Copernicus Marine Servvice In Situ TAC data model for 
    NRT products.
    
    The function returns an xarray dataset compliant with the Copernicus Marine Service 
    In Situ TAC data model for MY products.
    
    INPUT:
        tDS: xarray DataSet containing temporally aggregated total data.  
        radSite: DataFrame containing the information of the radial site that produced the radial
        
    OUTPUT:
        instacDS: xarray dataset compliant with the Copernicus Marine Service In Situ TAC data model for MY products
    """
    
    # Get data packing information per variable
    f = open('Data_Models/CMEMS_IN_SITU_TAC/Totals/Total_Data_Packing.json')
    dataPacking = json.loads(f.read())
    f.close()
    
    # Get variable attributes
    f = open('Data_Models/CMEMS_IN_SITU_TAC/Totals/Total_Variables.json')
    totVariables = json.loads(f.read())
    f.close()
    
    # Get global attributes
    f = open('Data_Models/CMEMS_IN_SITU_TAC/Global_Attributes.json')
    globalAttributes = json.loads(f.read())
    f.close()
    
    # Create the output dataset
    instacDS = tDS
    instacDS.encoding = {}
    
    # Evaluate time coverage start, end, resolution and duration
    timeCoverageStart = pd.Timestamp(instacDS['TIME'].values.min()).to_pydatetime() - relativedelta(minutes=networkData.iloc[0]['temporal_resolution']/2)
    timeCoverageEnd = pd.Timestamp(instacDS['TIME'].values.max()).to_pydatetime() + relativedelta(minutes=networkData.iloc[0]['temporal_resolution']/2)
    
    timeCoverageDuration = pd.Timedelta(timeCoverageEnd - timeCoverageStart).isoformat()
    
    # Build the file id
    ID = 'GL_TV_HF_' + tDS.attrs['platform_code'] + '_' + timeCoverageStart.strftime('%Y%m%d') + '-' + timeCoverageEnd.strftime('%Y%m%d')
    
    # Get the attributes and the data type of crs variable
    crsAttrs = instacDS.crs.attrs
    crsDataType = instacDS.crs.encoding['dtype']
    
    # Remove crs variable (it's time-varying because of the temporal aggregation)
    instacDS = instacDS.drop_vars('crs')
    
    # Add time-independent crs variable
    instacDS['crs'] = xr.DataArray(int(0), )
    instacDS['crs'].attrs = crsAttrs
    instacDS['crs'].encoding['dtype'] = crsDataType
    
    # Remove encoding for data variables
    for vv in instacDS:
        if 'char_dim_name' in instacDS[vv].encoding.keys():
            instacDS[vv] = instacDS[vv].astype(tDS[vv].encoding['char_dim_name'].replace('STRING','S'))
            instacDS[vv].encoding = {'char_dim_name': tDS[vv].encoding['char_dim_name']}
        else:
            instacDS[vv].encoding = {}
            
    # Add data variable attributes to the DataSet
    for vv in instacDS:
        instacDS[vv].attrs = totVariables[vv]
        
    # Add coordinate variable attributes to the DataSet
    for cc in instacDS.coords:
        instacDS[cc].attrs = totVariables[cc]
    
    # Modify data_mode variable attribute for data variables
    for vv in instacDS:
        if 'data_mode' in instacDS[vv].attrs:
            instacDS[vv].attrs['data_mode'] = 'D'
            
    # Modify data_mode variable attribute for coordinate variables
    for cc in instacDS.coords:
        if 'data_mode' in instacDS[cc].attrs:
            instacDS[cc].attrs['data_mode'] = 'D'
            
    # Update QC variable attribute "comment" for inserting test thresholds
    for qcv in list(tDS.keys()):
        if 'QC' in qcv:
            if not qcv in ['TIME_QC', 'POSITION_QC', 'DEPTH_QC']:
                instacDS[qcv].attrs['comment'] = instacDS[qcv].attrs['comment'] + ' ' + tDS[qcv].attrs['comment']   
    
    # Update QC variable attribute "flag_values" for assigning the right data type
    for qcv in instacDS:
        if 'QC' in qcv:
            instacDS[qcv].attrs['flag_values'] = list(np.int_(instacDS[qcv].attrs['flag_values']).astype(dataPacking[qcv]['dtype']))
        
    # Modify some global attributes
    instacDS.attrs['id'] = ID
    instacDS.attrs['data_mode'] = 'D'
    instacDS.attrs['time_coverage_start'] = timeCoverageStart.strftime('%Y-%m-%dT%H:%M:%SZ')
    instacDS.attrs['time_coverage_end'] = timeCoverageEnd.strftime('%Y-%m-%dT%H:%M:%SZ')
    instacDS.attrs['time_coverage_duration'] = timeCoverageDuration    
    creationDate = dt.datetime.utcnow()
    instacDS.attrs['date_created'] = creationDate.strftime('%Y-%m-%dT%H:%M:%SZ')
    instacDS.attrs['date_modified'] = creationDate.strftime('%Y-%m-%dT%H:%M:%SZ')
    instacDS.attrs['history'] = 'Data measured from ' + timeCoverageStart.strftime('%Y-%m-%dT%H:%M:%SZ') + ' to ' \
                                + timeCoverageEnd.strftime('%Y-%m-%dT%H:%M:%SZ') + '. netCDF file created at ' \
                                + creationDate.strftime('%Y-%m-%dT%H:%M:%SZ') + ' by the European HFR Node.'        
    
    # Encode data types, data packing and _FillValue for the data variables of the DataSet
    for vv in instacDS:
        if vv in dataPacking:
            if 'dtype' in dataPacking[vv]:
                instacDS[vv].encoding['dtype'] = dataPacking[vv]['dtype']
            if 'scale_factor' in dataPacking[vv]:
                instacDS[vv].encoding['scale_factor'] = dataPacking[vv]['scale_factor']    
            if 'add_offset' in dataPacking[vv]:
                instacDS[vv].encoding['add_offset'] = dataPacking[vv]['add_offset']
            if 'fill_value' in dataPacking[vv]:
                if not vv in ['SCDR', 'SCDT']:
                    instacDS[vv].encoding['_FillValue'] = nc4.default_fillvals[np.dtype(dataPacking[vv]['dtype']).kind + str(np.dtype(dataPacking[vv]['dtype']).itemsize)]
                else:
                    instacDS[vv].encoding['_FillValue'] = b' '
                    
            else:
                instacDS[vv].encoding['_FillValue'] = None
                
    # Update valid_min and valid_max variable attributes according to data packing for data variables
    for vv in instacDS:
        if 'valid_min' in totVariables[vv]:
            if ('scale_factor' in dataPacking[vv]) and ('add_offset' in dataPacking[vv]):
                instacDS[vv].attrs['valid_min'] = np.float_(((totVariables[vv]['valid_min'] - dataPacking[vv]['add_offset']) / dataPacking[vv]['scale_factor'])).astype(dataPacking[vv]['dtype'])
            else:
                instacDS[vv].attrs['valid_min'] = np.float_(totVariables[vv]['valid_min']).astype(dataPacking[vv]['dtype'])
        if 'valid_max' in totVariables[vv]:             
            if ('scale_factor' in dataPacking[vv]) and ('add_offset' in dataPacking[vv]):
                instacDS[vv].attrs['valid_max'] = np.float_(((totVariables[vv]['valid_max'] - dataPacking[vv]['add_offset']) / dataPacking[vv]['scale_factor'])).astype(dataPacking[vv]['dtype'])
            else:
                instacDS[vv].attrs['valid_max'] = np.float_(totVariables[vv]['valid_max']).astype(dataPacking[vv]['dtype'])
                
    # Encode data types, data packing and _FillValue for the coordinate variables of the DataSet
    for cc in instacDS.coords:
        if cc in dataPacking:
            if 'dtype' in dataPacking[cc]:
                instacDS[cc].encoding['dtype'] = dataPacking[cc]['dtype']
            if 'scale_factor' in dataPacking[cc]:
                instacDS[cc].encoding['scale_factor'] = dataPacking[cc]['scale_factor']                
            if 'add_offset' in dataPacking[cc]:
                instacDS[cc].encoding['add_offset'] = dataPacking[cc]['add_offset']
            if 'fill_value' in dataPacking[cc]:
                instacDS[cc].encoding['_FillValue'] = nc4.default_fillvals[np.dtype(dataPacking[cc]['dtype']).kind + str(np.dtype(dataPacking[cc]['dtype']).itemsize)]
            else:
                instacDS[cc].encoding['_FillValue'] = None
        
    # Update valid_min and valid_max variable attributes according to data packing for coordinate variables
    for cc in instacDS.coords:
        if 'valid_min' in totVariables[cc]:
            if ('scale_factor' in dataPacking[cc]) and ('add_offset' in dataPacking[cc]):
                instacDS[cc].attrs['valid_min'] = np.float_(((totVariables[cc]['valid_min'] - dataPacking[cc]['add_offset']) / dataPacking[cc]['scale_factor'])).astype(dataPacking[cc]['dtype'])
            else:
                instacDS[cc].attrs['valid_min'] = np.float_(totVariables[cc]['valid_min']).astype(dataPacking[cc]['dtype'])
        if 'valid_max' in totVariables[cc]:             
            if ('scale_factor' in dataPacking[cc]) and ('add_offset' in dataPacking[cc]):
                instacDS[cc].attrs['valid_max'] = np.float_(((totVariables[cc]['valid_max'] - dataPacking[cc]['add_offset']) / dataPacking[cc]['scale_factor'])).astype(dataPacking[cc]['dtype'])
            else:
                instacDS[cc].attrs['valid_max'] = np.float_(totVariables[cc]['valid_max']).astype(dataPacking[cc]['dtype'])
           
    return instacDS

def adjustToMYINSTACradialDatamodel(rDS, radSite):
    """
    This function adjusts the data model of the input aggregated radial dataset for complying with
    the Copernicus Marine Service data model for MY products.
    the input dataset must follow the Copernicus Marine Servvice In Situ TAC data model for 
    NRT products.
    
    The function returns an xarray dataset compliant with the Copernicus Marine Service 
    In Situ TAC data model for MY products.
    
    INPUT:
        rDS: xarray DataSet containing temporally aggregated radial data.  
        radSite: DataFrame containing the information of the radial site that produced the radial
        
    OUTPUT:
        instacDS: xarray dataset compliant with the Copernicus Marine Service In Situ TAC data model for MY products
    """
    
    # Get data packing information per variable
    f = open('Data_Models/CMEMS_IN_SITU_TAC/Radials/Radial_Data_Packing.json')
    dataPacking = json.loads(f.read())
    f.close()
    
    # Get variable attributes
    f = open('Data_Models/CMEMS_IN_SITU_TAC/Radials/Radial_Variables.json')
    radVariables = json.loads(f.read())
    f.close()
    
    # Get global attributes
    f = open('Data_Models/CMEMS_IN_SITU_TAC/Global_Attributes.json')
    globalAttributes = json.loads(f.read())
    f.close()
    
    # Create the output dataset
    instacDS = rDS
    instacDS.encoding = {}
    
    # Evaluate time coverage start, end, resolution and duration
    timeCoverageStart = pd.Timestamp(instacDS['TIME'].values.min()).to_pydatetime() - relativedelta(minutes=radSite.iloc[0]['temporal_resolution']/2)
    timeCoverageEnd = pd.Timestamp(instacDS['TIME'].values.max()).to_pydatetime() + relativedelta(minutes=radSite.iloc[0]['temporal_resolution']/2)
    
    timeCoverageDuration = pd.Timedelta(timeCoverageEnd - timeCoverageStart).isoformat()
    
    # Build the file id
    ID = 'GL_RV_HF_' + rDS.attrs['platform_code'] + '_' + timeCoverageStart.strftime('%Y%m%d') + '-' + timeCoverageEnd.strftime('%Y%m%d')
    
    # Get the attributes and the data type of crs variable
    crsAttrs = instacDS.crs.attrs
    crsDataType = instacDS.crs.encoding['dtype']
    
    # Remove crs variable (it's time-varying because of the temporal aggregation)
    instacDS = instacDS.drop_vars('crs')
    
    # Add time-independent crs variable
    instacDS['crs'] = xr.DataArray(int(0), )
    instacDS['crs'].attrs = crsAttrs
    instacDS['crs'].encoding['dtype'] = crsDataType
    
    # Remove encoding for data variables
    for vv in instacDS:
        if 'char_dim_name' in instacDS[vv].encoding.keys():
            instacDS[vv] = instacDS[vv].astype(rDS[vv].encoding['char_dim_name'].replace('STRING','S'))
            instacDS[vv].encoding = {'char_dim_name': rDS[vv].encoding['char_dim_name']}
        else:
            instacDS[vv].encoding = {}
            
    # Add data variable attributes to the DataSet
    for vv in instacDS:
        instacDS[vv].attrs = radVariables[vv]
        
    # Add coordinate variable attributes to the DataSet
    for cc in instacDS.coords:
        instacDS[cc].attrs = radVariables[cc]
    
    # Modify data_mode variable attribute for data variables
    for vv in instacDS:
        if 'data_mode' in instacDS[vv].attrs:
            instacDS[vv].attrs['data_mode'] = 'D'
            
    # Modify data_mode variable attribute for coordinate variables
    for cc in instacDS.coords:
        if 'data_mode' in instacDS[cc].attrs:
            instacDS[cc].attrs['data_mode'] = 'D'
            
    # Update QC variable attribute "comment" for inserting test thresholds
    for qcv in list(rDS.keys()):
        if 'QC' in qcv:
            if not qcv in ['TIME_QC', 'POSITION_QC', 'DEPTH_QC']:
                instacDS[qcv].attrs['comment'] = instacDS[qcv].attrs['comment'] + ' ' + rDS[qcv].attrs['comment']   
    
    # Update QC variable attribute "flag_values" for assigning the right data type
    for qcv in instacDS:
        if 'QC' in qcv:
            instacDS[qcv].attrs['flag_values'] = list(np.int_(instacDS[qcv].attrs['flag_values']).astype(dataPacking[qcv]['dtype']))
        
    # Modify some global attributes
    instacDS.attrs['id'] = ID
    instacDS.attrs['data_mode'] = 'D'
    instacDS.attrs['time_coverage_start'] = timeCoverageStart.strftime('%Y-%m-%dT%H:%M:%SZ')
    instacDS.attrs['time_coverage_end'] = timeCoverageEnd.strftime('%Y-%m-%dT%H:%M:%SZ')
    instacDS.attrs['time_coverage_duration'] = timeCoverageDuration    
    creationDate = dt.datetime.utcnow()
    instacDS.attrs['date_created'] = creationDate.strftime('%Y-%m-%dT%H:%M:%SZ')
    instacDS.attrs['date_modified'] = creationDate.strftime('%Y-%m-%dT%H:%M:%SZ')
    instacDS.attrs['history'] = 'Data measured from ' + timeCoverageStart.strftime('%Y-%m-%dT%H:%M:%SZ') + ' to ' \
                                + timeCoverageEnd.strftime('%Y-%m-%dT%H:%M:%SZ') + '. netCDF file created at ' \
                                + creationDate.strftime('%Y-%m-%dT%H:%M:%SZ') + ' by the European HFR Node.'        
    
    # Encode data types, data packing and _FillValue for the data variables of the DataSet
    for vv in instacDS:
        if vv in dataPacking:
            if 'dtype' in dataPacking[vv]:
                instacDS[vv].encoding['dtype'] = dataPacking[vv]['dtype']
            if 'scale_factor' in dataPacking[vv]:
                instacDS[vv].encoding['scale_factor'] = dataPacking[vv]['scale_factor']                
            if 'add_offset' in dataPacking[vv]:
                instacDS[vv].encoding['add_offset'] = dataPacking[vv]['add_offset']
            if 'fill_value' in dataPacking[vv]:
                if not vv in ['SCDR', 'SCDT']:
                    instacDS[vv].encoding['_FillValue'] = nc4.default_fillvals[np.dtype(dataPacking[vv]['dtype']).kind + str(np.dtype(dataPacking[vv]['dtype']).itemsize)]
                else:
                    instacDS[vv].encoding['_FillValue'] = b' '
                    
            else:
                instacDS[vv].encoding['_FillValue'] = None
                
    # Update valid_min and valid_max variable attributes according to data packing for data variables
    for vv in instacDS:
        if 'valid_min' in radVariables[vv]:
            if ('scale_factor' in dataPacking[vv]) and ('add_offset' in dataPacking[vv]):
                instacDS[vv].attrs['valid_min'] = np.float_(((radVariables[vv]['valid_min'] - dataPacking[vv]['add_offset']) / dataPacking[vv]['scale_factor'])).astype(dataPacking[vv]['dtype'])
            else:
                instacDS[vv].attrs['valid_min'] = np.float_(radVariables[vv]['valid_min']).astype(dataPacking[vv]['dtype'])
        if 'valid_max' in radVariables[vv]:             
            if ('scale_factor' in dataPacking[vv]) and ('add_offset' in dataPacking[vv]):
                instacDS[vv].attrs['valid_max'] = np.float_(((radVariables[vv]['valid_max'] - dataPacking[vv]['add_offset']) / dataPacking[vv]['scale_factor'])).astype(dataPacking[vv]['dtype'])
            else:
                instacDS[vv].attrs['valid_max'] = np.float_(radVariables[vv]['valid_max']).astype(dataPacking[vv]['dtype'])
                
    # Encode data types, data packing and _FillValue for the coordinate variables of the DataSet
    for cc in instacDS.coords:
        if cc in dataPacking:
            if 'dtype' in dataPacking[cc]:
                instacDS[cc].encoding['dtype'] = dataPacking[cc]['dtype']
            if 'scale_factor' in dataPacking[cc]:
                instacDS[cc].encoding['scale_factor'] = dataPacking[cc]['scale_factor']                
            if 'add_offset' in dataPacking[cc]:
                instacDS[cc].encoding['add_offset'] = dataPacking[cc]['add_offset']
            if 'fill_value' in dataPacking[cc]:
                instacDS[cc].encoding['_FillValue'] = nc4.default_fillvals[np.dtype(dataPacking[cc]['dtype']).kind + str(np.dtype(dataPacking[cc]['dtype']).itemsize)]
            else:
                instacDS[cc].encoding['_FillValue'] = None
        
    # Update valid_min and valid_max variable attributes according to data packing for coordinate variables
    for cc in instacDS.coords:
        if 'valid_min' in radVariables[cc]:
            if ('scale_factor' in dataPacking[cc]) and ('add_offset' in dataPacking[cc]):
                instacDS[cc].attrs['valid_min'] = np.float_(((radVariables[cc]['valid_min'] - dataPacking[cc]['add_offset']) / dataPacking[cc]['scale_factor'])).astype(dataPacking[cc]['dtype'])
            else:
                instacDS[cc].attrs['valid_min'] = np.float_(radVariables[cc]['valid_min']).astype(dataPacking[cc]['dtype'])
        if 'valid_max' in radVariables[cc]:             
            if ('scale_factor' in dataPacking[cc]) and ('add_offset' in dataPacking[cc]):
                instacDS[cc].attrs['valid_max'] = np.float_(((radVariables[cc]['valid_max'] - dataPacking[cc]['add_offset']) / dataPacking[cc]['scale_factor'])).astype(dataPacking[cc]['dtype'])
            else:
                instacDS[cc].attrs['valid_max'] = np.float_(radVariables[cc]['valid_max']).astype(dataPacking[cc]['dtype'])
           
    return instacDS

def aggregateTotals(groupedTot,networkData,instacFolder,outputFolder,monthly,yearly,history,compression,vers,logger):
    """
    This function performs the temporal aggregation of radial files into the MY netCDF files
    according to the Copernicus Marine Service In Situ TAC data model.
    The aggregation is performed according to the time interval specified in input.
    
    INPUTS:
        groupedTot: DataFrame containing the radials to be processed grouped by timestamp
                    for the input network with the related information
        networkData: DataFRame containing the information about the network to be processed
        instacFolder: full path of the folder where to pick input data for Copernicus Marine Service 
        outputFolder: full path of the folder where to save output data for Copernicus Marine Service 
        monthly: boolean for enabling monthly aggregation
        yearly: boolean for enabling yearly aggregation
        history: boolean for enabling the aggregation of all present files
        compression: boolean for enabling netCDF compression
        vers: version of the data model
        logger: logger object of the current processing

        
    OUTPUTS:
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    aTerr = False
    
    try:
        
        #####
        # Manage the output folders
        #####
        
        # Get station id
        networkID = groupedTot.iloc[0]['network_id']
        
        # Set the output folder path
        outputFolder = os.path.join(outputFolder,'MY',networkID,'Totals',vers)
        if not os.path.isdir(outputFolder):
            os.makedirs(outputFolder)         
        
        #####        
        # Open the netCDF files in an aggregated dataset
        #####
        
        # Create the list of the total files to be aggregated
        totalFiles = [os.path.join(groupedTot.iloc[idx]['filepath'],groupedTot.iloc[idx]['filename']) for idx in np.arange(len(groupedTot))]

        if len(totalFiles)>0:
            # Open all netCDF files to be aggregated
            aggrDS = xr.open_mfdataset(totalFiles,combine='nested',concat_dim='TIME',coords='minimal',compat='override',join='override')
            # aggrDS = xr.open_mfdataset(totalFiles,combine='nested',concat_dim='TIME',join='override')
            
        #####        
        # Convert to Copernicus Marine Service In Situ TAC data format for MY products
        #####
            
            # Apply the Copernicus Marine Service In Situ TAC data model
            myDS = adjustToMYINSTACtotalDatamodel(aggrDS,networkData)
            
            # Perform compression, if needed
            if compression:
                enc = {}
                for vv in myDS.data_vars:
                    if myDS[vv].ndim < 2:
                        continue                
                    enc[vv] = myDS[vv].encoding
                    enc[vv]['zlib'] = True
                    enc[vv]['complevel'] = 9
                    enc[vv]['fletcher32'] = True
            
            # Set the filename (with full path) for the aggregated netCDF file
            if monthly:
                timeStr = '_' + pd.Timestamp(myDS.TIME.values.min()).to_pydatetime().strftime('%Y%m')
            elif yearly:
                timeStr = '_' + pd.Timestamp(myDS.TIME.values.min()).to_pydatetime().strftime('%Y')
            elif history:
                timeStr = ''
            else:
                timeStr = '_' + pd.Timestamp(myDS.TIME.values.min()).to_pydatetime().strftime('%Y%m%d') + '-' + pd.Timestamp(myDS.TIME.values.max()).to_pydatetime().strftime('%Y%m%d')
            ncFilenameInstac = 'GL_TV_HF_' + networkID + '_Total' + timeStr + '.nc'
            ncFileInstac = os.path.join(outputFolder, ncFilenameInstac )
            
            # Modify ID global attribute if needed
            if monthly or yearly :
                myDS.attrs['id'] = 'GL_TV_HF_' + myDS.attrs['platform_code'] + '_' + timeStr
            elif history:
                myDS.attrs['id'] = 'GL_TV_HF_' + myDS.attrs['platform_code']
            
            # Check if the netCDF file exists and remove it
            if os.path.isfile(ncFileInstac):
                os.remove(ncFileInstac)
            
            # Create netCDF from DataSet and save it
            if compression:
                myDS.to_netcdf(ncFileInstac, format='NETCDF4_CLASSIC', engine='netcdf4', encoding=enc)
            else:
                myDS.to_netcdf(ncFileInstac, format='NETCDF4_CLASSIC', engine='netcdf4')
            
            # Modify the units attribute of TIME variable for including timezone digit
            ncf = nc4.Dataset(ncFileInstac,'r+',format='NETCDF4_CLASSIC')
            ncf.variables['TIME'].units = 'days since 1950-01-01T00:00:00Z'
            ncf.variables['TIME'].calendar = 'standard'
            ncf.close()
            
            # Remove 'coordinates' attribute from variables POSITION_QC, QCflag, VART_QC, GDOP_QC, DDNS_QC, CSPD_QC
            ncf = nc4.Dataset(ncFileInstac,'r+',format='NETCDF4_CLASSIC')
            ncf.variables['POSITION_QC'].delncattr('coordinates')
            ncf.variables['QCflag'].delncattr('coordinates')
            ncf.variables['VART_QC'].delncattr('coordinates')
            ncf.variables['GDOP_QC'].delncattr('coordinates')
            ncf.variables['DDNS_QC'].delncattr('coordinates')
            ncf.variables['CSPD_QC'].delncattr('coordinates')
            ncf.close()
            
            logger.info(ncFilenameInstac + ' total netCDF file succesfully created and stored in Copericus Marine Service In Situ TAC buffer (' + vers + ').')
            
        else:
            return
        
    except Exception as err:
        aTerr = True
        if 'ncFilenameInstac' in locals():
            logger.error(err.args[0] + ' in creating Copernicus Marine Service In Situ TAC radial file ' + ncFilenameInstac)
        else:
            logger.error(err.args[0] + ' in creating Copernicus Marine Service In Situ TAC radial file for time interval ' + timeStr)
        return     
    
    return  

    
def aggregateRadials(groupedRad,networkID,stationData,instacFolder,outputFolder,monthly,yearly,history,compression,vers,logger):
    """
    This function performs the temporal aggregation of radial files into the MY netCDF files
    according to the Copernicus Marine Service In Situ TAC data model.
    The aggregation is performed according to the time interval specified in input.
    
    INPUTS:
        groupedRad: DataFrame containing the radials to be processed grouped by timestamp
                    for the input network with the related information
        networkID: network ID of the network to be processed
        stationData: DataFrame containing the information of the stations belonging 
                     to the network to be processed
        instacFolder: full path of the folder where to pick input data for Copernicus Marine Service 
        outputFolder: full path of the folder where to save output data for Copernicus Marine Service 
        monthly: boolean for enabling monthly aggregation
        yearly: boolean for enabling yearly aggregation
        history: boolean for enabling the aggregation of all present files
        compression: boolean for enabling netCDF compression
        vers: version of the data model
        logger: logger object of the current processing

        
    OUTPUTS:
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    aRerr = False
    
    try:
        
        #####
        # Manage the output folders
        #####
        
        # Get station id
        stationID = groupedRad.iloc[0]['station_id']
        
        # Set the output folder path
        outputFolder = os.path.join(outputFolder,'MY',networkID,'Radials',vers,stationID)
        if not os.path.isdir(outputFolder):
            os.makedirs(outputFolder)         
        
        #####        
        # Open the netCDF files in an aggregated dataset
        #####
        
        # Create the list of the radial files to be aggregated
        radialFiles = [os.path.join(groupedRad.iloc[idx]['filepath'],groupedRad.iloc[idx]['filename']) for idx in np.arange(len(groupedRad))]

        if len(radialFiles)>0:
            # Open all netCDF files to be aggregated
            aggrDS = xr.open_mfdataset(radialFiles,combine='nested',concat_dim='TIME',coords='minimal',compat='override',join='override')
            
        #####        
        # Convert to Copernicus Marine Service In Situ TAC data format for MY products
        #####
            
            # Apply the Copernicus Marine Service In Situ TAC data model
            myDS = adjustToMYINSTACradialDatamodel(aggrDS,stationData.loc[stationData['station_id'] == stationID])
            
            # Perform compression, if needed
            if compression:
                enc = {}
                for vv in myDS.data_vars:
                    if myDS[vv].ndim < 2:
                        continue                
                    enc[vv] = myDS[vv].encoding
                    enc[vv]['zlib'] = True
                    enc[vv]['complevel'] = 9
                    enc[vv]['fletcher32'] = True
            
            # Set the filename (with full path) for the aggregated netCDF file
            if monthly:
                timeStr = '_' + pd.Timestamp(myDS.TIME.values.min()).to_pydatetime().strftime('%Y%m')
            elif yearly:
                timeStr = '_' + pd.Timestamp(myDS.TIME.values.min()).to_pydatetime().strftime('%Y')
            elif history:
                timeStr = ''
            else:
                timeStr = '_' + pd.Timestamp(myDS.TIME.values.min()).to_pydatetime().strftime('%Y%m%d') + '-' + pd.Timestamp(myDS.TIME.values.max()).to_pydatetime().strftime('%Y%m%d')
            ncFilenameInstac = 'GL_RV_HF_' + networkID + '-' + stationID + timeStr + '.nc'
            ncFileInstac = os.path.join(outputFolder, ncFilenameInstac )
            
            # Modify ID global attribute if needed
            if monthly or yearly :
                myDS.attrs['id'] = 'GL_RV_HF_' + myDS.attrs['platform_code'] + '_' + timeStr
            elif history:
                myDS.attrs['id'] = 'GL_RV_HF_' + myDS.attrs['platform_code']
            
            # Check if the netCDF file exists and remove it
            if os.path.isfile(ncFileInstac):
                os.remove(ncFileInstac)
            
            # Create netCDF from DataSet and save it
            if compression:
                myDS.to_netcdf(ncFileInstac, format='NETCDF4_CLASSIC', engine='netcdf4', encoding=enc)
            else:
                myDS.to_netcdf(ncFileInstac, format='NETCDF4_CLASSIC', engine='netcdf4')
            
            # Modify the units attribute of TIME variable for including timezone digit
            ncf = nc4.Dataset(ncFileInstac,'r+',format='NETCDF4_CLASSIC')
            ncf.variables['TIME'].units = 'days since 1950-01-01T00:00:00Z'
            ncf.variables['TIME'].calendar = 'standard'
            ncf.close()
            
            logger.info(ncFilenameInstac + ' radial netCDF file succesfully created and stored in Copericus Marine Service In Situ TAC buffer (' + vers + ').')
            
        else:
            return
        
    except Exception as err:
        aRerr = True
        if 'ncFilenameInstac' in locals():
            logger.error(err.args[0] + ' in creating Copernicus Marine Service In Situ TAC radial file ' + ncFilenameInstac)
        else:
            logger.error(err.args[0] + ' in creating Copernicus Marine Service In Situ TAC radial file for time interval ' + timeStr)
        return     
    
    return  


def selectTotals(networkID, startDate, endDate, instacFolder, vers, logger):
    """
    This function lists the daily aggregated total files produced for NRT products
    that falls into the aggregation time interval and creates the DataFrame containing 
    the information needed for the aggregation of total files into the MY netCDF files
    according to the Copernicus Marine Service In Situ TAC data model.
    
    INPUTS:
        networkID: network ID of the network to be processed
        startDate: datetime of the initial date of the processing period
        endDate: datetime of the final date of the processing period
        instacFolder: full path of the folder where to save data for Copernicus Marine Service 
        vers: version of the data model
        logger: logger object of the current processing

        
    OUTPUTS:
        totalsToBeAggregated: DataFrame containing all the totals to be aggregated for the input 
                              network with the related information
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    sTerr = False
    
    # Create output Series
    totalsToBeAggregated = pd.DataFrame(columns=['filename', 'filepath', 'network_id', 'datetime'])
    
    #####
    # List totals from network
    #####
    
    try:   
        logger.info('Total selection started for ' + networkID + ' network.')
        # Set input folder path
        inputFolder = os.path.join(instacFolder, networkID, 'Totals', vers)
        
        # Check if the input folder path exists
        if not os.path.isdir(inputFolder):
            logger.info('The total input folder for network ' + networkID + ' does not exist.')
        else:
            # List all total files
            inputFiles = [file for file in glob.glob(os.path.join(inputFolder,'**/GL_TV_HF_' + networkID + '-Total*.nc'), recursive = True)]                    
            inputFiles.sort()
            for inputFile in inputFiles:
                try:
                    # Get file parts
                    filePath = os.path.dirname(inputFile)
                    fileName = os.path.basename(inputFile)
                    fileExt = os.path.splitext(inputFile)[1]
                    
                    # Get file timestamp
                    dateTime = dt.datetime.strptime(fileName.split('_')[-1].split('.')[0], "%Y%m%d").date()
                        
    #####
    # Insert total information into the output DataFrame
    #####

                    # Check if a time interval for aggregation is specified
                    if startDate:    
                        # Check if the total falls into the processing time interval
                        if ((dateTime >= startDate) and (dateTime <= endDate)):    
                            # Prepare data to be inserted into the output DataFrame
                            dataTotal = {'filename': [fileName], 'filepath': [filePath], 'network_id': [networkID], 'datetime': [dateTime]}
                            dfTotal = pd.DataFrame(dataTotal)
                            # Insert into the output DataFrame
                            totalsToBeAggregated = pd.concat([totalsToBeAggregated, dfTotal],ignore_index=True)
                    else:
                        # Prepare data to be inserted into the output DataFrame
                        dataTotal = {'filename': [fileName], 'filepath': [filePath], 'network_id': [networkID], 'datetime': [dateTime]}
                        dfTotal = pd.DataFrame(dataTotal)                                
                        # Insert into the output DataFrame
                        totalsToBeAggregated = pd.concat([totalsToBeAggregated, dfTotal],ignore_index=True)

                except Exception as err:
                    sTerr = True
                    logger.error(err.args[0] + ' for file ' + fileName)
                        
    except Exception as err:
        sTerr = True
        logger.error(err.args[0] + ' for network ' + networkID)
    
    return totalsToBeAggregated

def selectRadials(networkID, stationData, startDate, endDate, instacFolder, vers, logger):
    """
    This function lists the daily aggregated radial files produced for NRT products
    that falls into the aggregation time interval and creates the DataFrame containing 
    the information needed for the aggregation of radial files into the MY netCDF files
    according to the Copernicus Marine Service In Situ TAC data model.
    
    INPUTS:
        networkID: network ID of the network to be processed
        stationData: DataFrame containing the information of the stations belonging 
                     to the network to be processed
        startDate: datetime of the initial date of the processing period
        endDate: datetime of the final date of the processing period
        instacFolder: full path of the folder where to save data for Copernicus Marine Service 
        vers: version of the data model
        logger: logger object of the current processing

        
    OUTPUTS:
        radialsToBeAggregated: DataFrame containing all the radials to be aggregated for the input 
                              network with the related information
        
    """
    #####
    # Setup
    #####
    
    # Initialize error flag
    sRerr = False
    
    # Create output Series
    radialsToBeAggregated = pd.DataFrame(columns=['filename', 'filepath', 'network_id', 'station_id', 'datetime'])
    
    #####
    # List radials from stations
    #####
    
    # Scan stations
    for st in range(len(stationData)):
        try:   
            # Get station id
            stationID = stationData.iloc[st]['station_id']
            logger.info('Radial selection started for ' + networkID + '-' + stationID + ' station.')
            # Set input folder path
            inputFolder = os.path.join(instacFolder, networkID, 'Radials', vers, stationID)
            
            # Check if the input folder path exists
            if not os.path.isdir(inputFolder):
                logger.info('The radial input folder for station ' + networkID + '-' + stationID + ' does not exist.')
            else:
                # List all radial files
                inputFiles = [file for file in glob.glob(os.path.join(inputFolder,'**/GL_RV_HF_' + networkID + '-' + stationID + '*.nc'), recursive = True)]                    
                inputFiles.sort()
                for inputFile in inputFiles:
                    try:
                        # Get file parts
                        filePath = os.path.dirname(inputFile)
                        fileName = os.path.basename(inputFile)
                        fileExt = os.path.splitext(inputFile)[1]
                        
                        # Get file timestamp
                        dateTime = dt.datetime.strptime(fileName.split('_')[-1].split('.')[0], "%Y%m%d").date()
                            
    #####
    # Insert radial information into the output DataFrame
    #####
    
                        # Check if a time interval for aggregation is specified
                        if startDate:    
                            # Check if the radial falls into the processing time interval
                            if ((dateTime >= startDate) and (dateTime <= endDate)):    
                                # Prepare data to be inserted into the output DataFrame
                                dataRadial = {'filename': [fileName], 'filepath': [filePath], 'network_id': [networkID], \
                                              'station_id': [stationID], 'datetime': [dateTime]}
                                dfRadial = pd.DataFrame(dataRadial)
                                # Insert into the output DataFrame
                                radialsToBeAggregated = pd.concat([radialsToBeAggregated, dfRadial],ignore_index=True)
                        else:
                            # Prepare data to be inserted into the output DataFrame
                            dataRadial = {'filename': [fileName], 'filepath': [filePath], 'network_id': [networkID], \
                                          'station_id': [stationID], 'datetime': [dateTime]}
                            dfRadial = pd.DataFrame(dataRadial)                                
                            # Insert into the output DataFrame
                            radialsToBeAggregated = pd.concat([radialsToBeAggregated, dfRadial],ignore_index=True)

                    except Exception as err:
                        sRerr = True
                        logger.error(err.args[0] + ' for file ' + fileName)
                        
        except Exception as err:
            sRerr = True
            logger.error(err.args[0] + ' for station ' + stationID)
    
    return radialsToBeAggregated

def aggregateNetwork(networkID, startDate, endDate, monthly, yearly, history, compression, instacFolder, outputFolder, sqlConfig):
    """
    This function processes the radial and total files of a single HFR network
    for temporally aggregating netCDF files according to the Copernicus Marine 
    Service In-Situ TAC data model.
    
    The first processing step consists in selecting the netCDF files to be temporally aggregated,
    depending on the input options provided by the user.
    
    The second processing step consists in generating the temporally aggregated netCDF files 
    according to the Copernicus Marine Service In-Situ TAC data model for MY products.
    
    INPUTS:
        networkID: network ID of the network to be processed
        startDate: initial date of the processing period in datetime format
        endDate: final date of the processing period in datetime format
        monthly: boolean for enabling monthly aggregation
        yearly: boolean for enabling yearly aggregation
        history: boolean for enabling the aggregation of all present files
        compression: boolean for enabling netCDF compression
        instacFolder: full path of the folder where to pick input data for Copernicus Marine Service
        outputFolder: full path of the folder where to save output data for Copernicus Marine Service
        sqlConfig: parameters for connecting to the Mysql EU HFR NODE EU HFR NODE database

        
    OUTPUTS:
        aNerr: error flag (True = errors occurred, False = no error occurred)
        
    """
    #####
    # Setup
    #####
    
    # Set the version of the data model
    vers = 'v3'
    
    try:
        # Create the folder for the network log
        networkLogFolder = '/var/log/EU_HFR_NODE_MY/' + networkID
        if not os.path.isdir(networkLogFolder):
            os.mkdir(networkLogFolder)
               
        # Create logger
        logger = logging.getLogger('EU_HFR_NODE_MY_' + networkID)
        logger.setLevel(logging.INFO)
        # Create console handler and set level to DEBUG
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # Create logfile handler
        lfh = logging.FileHandler(networkLogFolder + '/EU_HFR_NODE_MY_' + networkID + '.log')
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
        aNerr = False
        
    except Exception as err:
        aNerr = True
        logger.error(err.args[0])
        logger.info('Exited with errors.')
        return aNerr
    
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
        if networkID != 'HFR-WesternItaly':
            stationSelectQuery = 'SELECT * FROM station_tb WHERE network_id=\'' + networkID + '\''
        stationData = pd.read_sql(stationSelectQuery, con=eng)
        numStations = stationData.shape[0]
        
        logger.info(networkID + ' station data successfully fetched from EU HFR NODE database.')
    except sqlalchemy.exc.DBAPIError as err:        
        aNerr = True
        logger.error('MySQL error ' + err._message())
        logger.info('Exited with errors.')
        return aNerr
    
    try:
        
    #####
    # Select HFR data
    #####
        
        # Select radials to be aggregated
        if ('HFR-US' in networkID) or ('HFR-WesternItaly' in networkID):
            pass
        else:
            radialsToBeAggregated = selectRadials(networkID,stationData,startDate,endDate,instacFolder,vers,logger)
            logger.info('Radials to be aggregated successfully selected for network ' + networkID)
        
        # Select totals to be aggregated
        totalsToBeAggregated = selectTotals(networkID,startDate,endDate,instacFolder,vers,logger)
        logger.info('Totals to be aggregated successfully selected for network ' + networkID)
        
    #####
    # Aggregate HFR data
    #####
        
        # Aggregate radials
        if ('HFR-US' in networkID) or ('HFR-WesternItaly' in networkID):
            pass
        else:            
            logger.info('Radial aggregation started for ' + networkID + ' network')
            if monthly:
                radialsToBeAggregated.index=pd.to_datetime(radialsToBeAggregated['datetime'])
                radialsToBeAggregated.groupby(by=['station_id', radialsToBeAggregated.index.year, radialsToBeAggregated.index.month], group_keys=False).apply(lambda x:aggregateRadials(x,networkID,stationData,instacFolder,outputFolder,monthly,yearly,history,compression,vers,logger))
            elif yearly:
                radialsToBeAggregated.index=pd.to_datetime(radialsToBeAggregated['datetime'])
                radialsToBeAggregated.groupby(by=['station_id', radialsToBeAggregated.index.year], group_keys=False).apply(lambda x:aggregateRadials(x,networkID,stationData,instacFolder,outputFolder,monthly,yearly,history,compression,vers,logger))
            else:
                radialsToBeAggregated.groupby('station_id', group_keys=False).apply(lambda x:aggregateRadials(x,networkID,stationData,instacFolder,outputFolder,monthly,yearly,history,compression,vers,logger))
        
        # Aggregate totals
        logger.info('Total aggregation started for ' + networkID + ' network')
        if monthly:
            totalsToBeAggregated.index=pd.to_datetime(totalsToBeAggregated['datetime'])
            totalsToBeAggregated.groupby(by=[totalsToBeAggregated.index.year, totalsToBeAggregated.index.month], group_keys=False).apply(lambda x:aggregateTotals(x,networkData,instacFolder,outputFolder,monthly,yearly,history,compression,vers,logger))
        elif yearly:
            totalsToBeAggregated.index=pd.to_datetime(totalsToBeAggregated['datetime'])
            totalsToBeAggregated.groupby(by=[totalsToBeAggregated.index.year], group_keys=False).apply(lambda x:aggregateTotals(x,networkData,instacFolder,outputFolder,monthly,yearly,history,compression,vers,logger))
        else:
            totalsToBeAggregated.groupby('network_id', group_keys=False).apply(lambda x:aggregateTotals(x,networkData,instacFolder,outputFolder,monthly,yearly,history,compression,vers,logger))
            
        # Wait a bit (useful for multiprocessing management)
        time.sleep(30)
            
    except Exception as err:
        aNerr = True
        logger.error(err.args[0])
        logger.info('Exited with errors.')
        return aNerr    
    
    return aNerr

####################
# MAIN DEFINITION
####################

def main(argv):
    
#####
# Setup
#####
       
    # Set the argument structure
    try:
        opts, args = getopt.getopt(argv,"n:s:e:myai:o:ch",["network=","start-date=","end-date=","monthly","yearly","all","instac-folder=","output-folder","compression","help"])
    except getopt.GetoptError:
        print('Usage: EU_HFR_NODE_MYaggregator.py -n <network ID of the network to be processed (if not specified, all the networks are processed)> ' \
              + '-s <initial date for processing formatted as yyyy-mm-dd (ISO8601 UTC date representation)> ' \
                  + '-e <final date for processing formatted as yyyy-mm-dd (ISO8601 UTC date representation)> ' \
                      + '-m <if specified enables the monthly aggregation (only one option among -m, -y, -a, [-s -e] must be specified)> ' \
                          + '-y <if specified enables the yearly aggregation (only one option among -m, -y, -a, [-s -e] must be specified)> ' \
                              + '-a <if specified enables the aggregation of all present files (only one option among -m, -y, -a, [-s -e] must be specified)> ' \
                                  + '-i <full path of the folder where to collect input data for Copernicus Marine Service>' \
                                      + '-o <full path of the folder where to save data for Copernicus Marine Service (if not specified, no files for Copernicus Marine Service are produced)>' \
                                          + '-c <if specified enables the compression of the output netCDF files (disabled by default)> ' \
                                              + '-h <shows help>')
        sys.exit(2)
        
    if not argv:
        print("No options specified. Please type 'EU_HFR_NODE_MYaggregator.py -h' for help.")
        sys.exit(2)
        
    if (('-m' in argv) or ('--monthly' in argv)):
        if (('-y' in argv) or ('--yearly' in argv)) or (('-a' in argv) or ('--all' in argv)) or (('-s' in argv) or ('--start-date' in argv)) or (('-e' in argv) or ('--end-date' in argv)):
            print("Too many aggregation intervals specified. Only one option among -m, -y, -a, [-s -e] must be specified. Please type 'EU_HFR_NODE_MYaggregator.py -h' for help.")
            sys.exit(2)
        
    if (('-y' in argv) or ('--yearly' in argv)):
        if (('-m' in argv) or ('--monthly' in argv)) or (('-a' in argv) or ('--all' in argv)) or (('-s' in argv) or ('--start-date' in argv)) or (('-e' in argv) or ('--end-date' in argv)):
            print("Too many aggregation intervals specified. Only one option among -m, -y, -a, [-s -e] must be specified. Please type 'EU_HFR_NODE_MYaggregator.py -h' for help.")
            sys.exit(2)
            
    if (('-a' in argv) or ('--all' in argv)):
        if (('-y' in argv) or ('--yearly' in argv)) or (('-m' in argv) or ('--monthly' in argv)) or (('-s' in argv) or ('--start-date' in argv)) or (('-e' in argv) or ('--end-date' in argv)):
            print("Too many aggregation intervals specified. Only one option among -m, -y, -a, [-s -e] must be specified. Please type 'EU_HFR_NODE_MYaggregator.py -h' for help.")
            sys.exit(2)
    
    if (('-s' in argv) or ('--start-date' in argv)) or (('-e' in argv) or ('--end-date' in argv)):
        if (('-y' in argv) or ('--yearly' in argv)) or (('-a' in argv) or ('--all' in argv)) or (('-m' in argv) or ('--monthly' in argv)):
            print("Too many aggregation intervals specified. Only one option among -m, -y, -a, [-s -e] must be specified. Please type 'EU_HFR_NODE_MYaggregator.py -h' for help.")
            sys.exit(2)
            
    if (('-s' in argv) or ('--start-date' in argv)) and (('-e' not in argv) and ('--end-date' not in argv)):
        print("No end date for aggregation specified. Please type 'EU_HFR_NODE_MYaggregator.py -h' for help.")
        sys.exit(2)
        
    if (('-e' in argv) or ('--end-date' in argv)) and (('-s' not in argv) and ('--start-date' not in argv)):
        print("No start date for aggregation specified. Please type 'EU_HFR_NODE_MYaggregator.py -h' for help.")
        sys.exit(2)
        
    if (('-i' not in argv) and ('--instac-folder' not in argv)):
        print("No input data folder specified. Please type 'EU_HFR_NODE_MYaggregator.py -h' for help.")
        sys.exit(2)
        
    if (('-o' not in argv) and ('--output-folder' not in argv)):
        print("No output data folder specified. Please type 'EU_HFR_NODE_MYaggregator.py -h' for help.")
        sys.exit(2)
        
    # Initialize optional arguments
    ntw = None
    startDate = None
    endDate = None
    monthly = False
    yearly = False
    history = False
    compression = False
        
    for opt, arg in opts:
        if opt == ("-h", "--help"):
            print('Usage: EU_HFR_NODE_MYaggregator.py -n <network ID of the network to be processed (if not specified, all the networks are processed)> ' \
                  + '-s <initial date for processing formatted as yyyy-mm-dd (ISO8601 UTC date representation)> ' \
                      + '-e <final date for processing formatted as yyyy-mm-dd (ISO8601 UTC date representation)> ' \
                          + '-m <if specified enables the monthly aggregation (only one option among -m, -y, -a, [-s -e] must be specified)> ' \
                              + '-y <if specified enables the yearly aggregation (only one option among -m, -y, -a, [-s -e] must be specified)> ' \
                                  + '-a <if specified enables the aggregation of all present files (only one option among -m, -y, -a, [-s -e] must be specified)> ' \
                                      + '-i <full path of the folder where to collect input data for Copernicus Marine Service>' \
                                          + '-o <full path of the folder where to save data for Copernicus Marine Service (if not specified, no files for Copernicus Marine Service are produced)>' \
                                              + '-c <if specified enables the compression of the output netCDF files (disabled by default)> ' \
                                                  + '-h <shows help>')
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
        elif opt in ("-m", "--monthly"):
            monthly = True
        elif opt in ("-y", "--yearly"):
            yearly = True
        elif opt in ("-a", "--all"):
            history = True            
        elif opt in ("-i", "--instac-folder"):
            instacFolder = arg.strip()
            # Check if the INSTAC folder path exists
            if not os.path.isdir(instacFolder):
                print('The specified folder for Copernicus Marine Service data does not exist.')
                sys.exit(2)
        elif opt in ("-o", "--output-folder"):
            outputFolder = arg.strip()
        elif opt in ("-c", "--compression"):
            compression = True
            
    # Check that initial date is before end date and convert them into date objects
    if startDate:
        startDate = startDate.date()
        endDate = endDate.date()
        if not startDate<endDate:
            print("Wrong time interval specified: initial date is later then end date")
            sys.exit(2)
          
    # Create logger
    logger = logging.getLogger('EU_HFR_NODE_MY')
    logger.setLevel(logging.INFO)
    # Create console handler and set level to DEBUG
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # Create logfile handler
    lfh = logging.FileHandler('/var/log/EU_HFR_NODE_MY/EU_HFR_NODE_MY.log')
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
    
    logger.info('MY temporal aggregation started.')
    
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
            networkSelectQuery = 'SELECT network_id FROM network_tb WHERE connected_to_REP=1'
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
        # Check if a specific network is selected for aggregation
        if ntw:
            aggregateNetwork(ntw, startDate, endDate, monthly, yearly, history, compression, instacFolder, outputFolder, sqlConfig)
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
                    pool[ex.submit(aggregateNetwork, ntw, startDate, endDate, monthly, yearly, history, compression, instacFolder, outputFolder, sqlConfig)] = ntw
                    logger.info('MY temporal aggregation for ' + ntw + ' network started')
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
                        logger.info('MY temporal aggregation for ' + trmNtw + ' network ended')
                        # Pop the process from the dictionary of running processes
                        pool.pop(future)
                        
                        # Check if networks waiting for processing are present in the queue
                        if networkQueue:
                            # Get the next network to be processed from the queue
                            nxtNtw = networkQueue[0]
                            
                            # Wait a bit (useful for multiprocessing management)
                            time.sleep(10)
                            # Start the process and insert process and the related network ID into the dictionary of the running processs
                            pool[ex.submit(aggregateNetwork, nxtNtw, startDate, endDate, monthly, yearly, history, compression, instacFolder, outputFolder, sqlConfig)] = nxtNtw
                            logger.info('MY temporal aggregation for ' + nxtNtw + ' network started')
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
    
    