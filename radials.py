import datetime as dt
from dateutil.relativedelta import relativedelta
import geopandas as gpd
import logging
import numpy as np
import pandas as pd
from pyproj import Geod, CRS
from collections import OrderedDict
import re
from shapely.geometry import Point
import xarray as xr
import netCDF4
from common import fileParser, create_dir, make_encoding
from calc import dms2dd, createLonLatGridFromBB, createLonLatGridFromBBwera, createLonLatGridFromTopLeftPointWera
import json
import warnings
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

# try:
#     # check for local configuration file
#     # user should copy configs_default to configs.py locally to change database login settings
#     from hfradar.configs.configs import netcdf_global_attributes
# except ModuleNotFoundError:
#     # if local configuration is not found, load the default
#     from hfradar.configs.configs_default import netcdf_global_attributes


# logger = logging.getLogger(__name__)


# def concatenate_multidimensional_radials(radial_list, enhance=False):
#     """
#     This function takes a list of Radial objects or radial file paths and
#     combines them along the time dimension using xarrays built-in concatenation
#     routines.
#     :param radial_list: list of radial files or Radial objects that you want to concatenate
#     :return: radials concatenated into an xarray dataset by range, bearing, and time
#     """

#     radial_dict = {}
#     for radial in radial_list:

#         if not isinstance(radial, Radial):
#             radial = Radial(radial)

#         radial_dict[radial.file_name] = radial.to_xarray_multidimensional(enhance=enhance)

#     ds = xr.concat(radial_dict.values(), 'time')
#     return ds.sortby('time')

def buildINSTACradialFilename(networkID,siteCode,ts,ext):
    """
    This function builds the filename for radial files according to 
    the structure used by Copernicus Marine Service In Situ TAC, 
    i.e. networkID-stationID_YYYY_MM_DD.
    
    INPUT:
        networkID: ID of the HFR network
        siteCode: code of the radial site
        ts: timestamp as datetime object
        ext: file extension
        
    OUTPUT:
        radFilename: filename for radial file.
    """
    # Get the time related part of the filename
    timeStr = ts.strftime("_%Y%m%d")
    
    # Build the filename
    radFilename = 'GL_RV_HF_' + networkID + '-' + siteCode + timeStr + ext
    
    return radFilename

def buildINSTACradialFolder(basePath,networkID,siteCode,vers):
    """
    This function builds the folder structure for storing radial files 
    according to the structure used by Copernicus Marine Service In Situ TAC, 
    i.e. networkID-stationID_YYYY_MM_DD.
    
    INPUT:
        basePath: base path
        networkID: ID of the HFR network
        siteCode: code of the radial site
        vers: version of the data model
        
    OUTPUT:
        radFolder: folder path for storing radial files.
    """
    # Strip trailing slash character
    basePath = basePath.rstrip('/')
    
    # Build the folder path
    radFolder = basePath + '/' + networkID + '/Radials/' + vers + '/' + siteCode + '/'
    
    return radFolder

def convertEHNtoINSTACradialDatamodel(rDS, networkData, stationData, version):
    """
    This function applies the Copernicus Marine Service data model to the input xarray
    dataset containing radial (either non temporally aggregated or temporally aggregated)
    data. The input dataset must follow the European standard data model.
    Variable data types and data packing information are collected from
    "Data_Models/CMEMS_IN_SITU_TAC/Radials/Radial_Data_Packing.json" file.
    Variable attribute schema is collected from 
    "Data_Models/CMEMS_IN_SITU_TAC/Radials/Radial_Variables.json" file.
    Global attribute schema is collected from 
    "Data_Models/CMEMS_IN_SITU_TAC/Global_Attributes.json" file.
    Global attributes are created starting from the input dataset and from 
    DataFrames containing the information about HFR network and radial station
    read from the EU HFR NODE database.
    The function returns an xarray dataset compliant with the Copernicus Marine Service 
    In Situ TAC data model.
    
    INPUT:
        rDS: xarray DataSet containing radial (either non temporally aggregated or temporally aggregated)
             data.
        networkData: DataFrame containing the information of the network to which the radial site belongs
        stationData: DataFrame containing the information of the radial site that produced the radial
        version: version of the data model        
        
    OUTPUT:
        instacDS: xarray dataset compliant with the Copernicus Marine Service In Situ TAC data model
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
    timeCoverageStart = pd.Timestamp(instacDS['TIME'].values.min()).to_pydatetime() - relativedelta(minutes=stationData.iloc[0]['temporal_resolution']/2)
    timeCoverageEnd = pd.Timestamp(instacDS['TIME'].values.max()).to_pydatetime() + relativedelta(minutes=stationData.iloc[0]['temporal_resolution']/2)
    
    timeCoverageDuration = pd.Timedelta(timeCoverageEnd - timeCoverageStart).isoformat()
    
    # Build the file id
    ID = 'GL_RV_HF_' + rDS.attrs['platform_code'] + '_' + pd.Timestamp(instacDS['TIME'].values.max()).to_pydatetime().strftime('%Y%m%d')
    
    # Get the TIME variable values
    xdsTime = instacDS.TIME.values
    # Convert them to datetime datetimes
    dtTime = np.array([pd.Timestamp(t).to_pydatetime() for t in xdsTime])
    
    # Evaluate timestamp as number of days since 1950-01-01T00:00:00Z
    timeDelta = dtTime - dt.datetime.strptime('1950-01-01T00:00:00Z','%Y-%m-%dT%H:%M:%SZ')
    ncTime = np.array([t.days + t.seconds / (60*60*24) for t in timeDelta])
    
    # Replace TIME variable values with timestamp as number of days since 1950-01-01T00:00:00Z
    instacDS = instacDS.assign_coords(TIME=ncTime)
    
    # Rename DEPTH_QC variable to DEPH_QC
    instacDS = instacDS.rename({'DEPTH_QC':'DEPH_QC'})  
    
    # Add DEPH variable
    instacDS['DEPH'] = xr.DataArray(0,
                             dims={'DEPTH': 1},
                             coords={'DEPTH': [0]})
    
    # Remove DEPTH variable
    instacDS = instacDS.drop_vars('DEPTH')
    
    # Remove crs variable (it's time-varying because of the temporal aggregation)
    instacDS = instacDS.drop_vars('crs')
    
    # Add time-independent crs variable
    instacDS['crs'] = xr.DataArray(int(0), )
    
    # Remove encoding for data variables
    for vv in instacDS:
        if 'char_dim_name' in instacDS[vv].encoding.keys():
            instacDS[vv].encoding = {'char_dim_name': instacDS[vv].encoding['char_dim_name']}
        else:
            instacDS[vv].encoding = {}
            
    # Remove attributes and encoding from coordinates variables
    for cc in instacDS.coords:
        instacDS[cc].attrs = {}
        instacDS[cc].encoding = {}
        
    # Add units and calendar to encoding for coordinate TIME
    instacDS['TIME'].encoding['units'] = 'days since 1950-01-01T00:00:00Z'
    instacDS['TIME'].encoding['calendar'] = 'standard'
    
    # Add data variable attributes to the DataSet
    for vv in instacDS:
        instacDS[vv].attrs = radVariables[vv]
        
    # Add coordinate variable attributes to the DataSet
    for cc in instacDS.coords:
        instacDS[cc].attrs = radVariables[cc]
    
    # Remove axis attributes from LONGITUDE and LATITUDE variables for polar geometry radials (i.e. Codar radials)
    if 'BEAR' in instacDS.coords:
        instacDS.LONGITUDE.attrs.pop('axis')
        instacDS.LATITUDE.attrs.pop('axis')
    
    # Update QC variable attribute "comment" for inserting test thresholds
    for qcv in list(rDS.keys()):
        if 'QC' in qcv:
            if not qcv in ['TIME_QC', 'POSITION_QC', 'DEPTH_QC']:
                instacDS[qcv].attrs['comment'] = instacDS[qcv].attrs['comment'] + ' ' + rDS[qcv].attrs['comment']   
    
    # Update QC variable attribute "flag_values" for assigning the right data type
    for qcv in instacDS:
        if 'QC' in qcv:
            instacDS[qcv].attrs['flag_values'] = list(np.int_(instacDS[qcv].attrs['flag_values']).astype(dataPacking[qcv]['dtype']))
        
    # Fill global attributes
    globalAttributes['site_code'] = rDS.attrs['site_code']
    globalAttributes['platform_code'] = rDS.attrs['platform_code']
    globalAttributes['platform_name'] = globalAttributes['platform_code']
    if 'wmo_platform_code' in rDS.attrs.keys():
        globalAttributes['wmo_platform_code'] = rDS.attrs['wmo_platform_code']
    globalAttributes['doa_estimation_method'] = rDS.attrs['doa_estimation_method']
    globalAttributes['calibration_type'] = rDS.attrs['calibration_type']
    globalAttributes['last_calibration_date'] = rDS.attrs['last_calibration_date']
    globalAttributes['calibration_link'] = rDS.attrs['calibration_link']
    globalAttributes['summary'] = rDS.attrs['summary']    
    globalAttributes['institution'] = rDS.attrs['institution']
    globalAttributes['institution_edmo_code'] = rDS.attrs['institution_edmo_code']
    globalAttributes['institution_references'] = rDS.attrs['institution_references']   
    # globalAttributes['institution_abreviated']
    # globalAttributes['institution_country']   
    globalAttributes['id'] = ID
    globalAttributes['project'] = rDS.attrs['project']
    globalAttributes['comment'] = rDS.attrs['comment']
    globalAttributes['network'] = rDS.attrs['network']
    globalAttributes['geospatial_lat_min'] = rDS.attrs['geospatial_lat_min']
    globalAttributes['geospatial_lat_max'] = rDS.attrs['geospatial_lat_max']
    globalAttributes['geospatial_lat_resolution'] = rDS.attrs['geospatial_lat_resolution']   
    globalAttributes['geospatial_lon_min'] = rDS.attrs['geospatial_lon_min']
    globalAttributes['geospatial_lon_max'] = rDS.attrs['geospatial_lon_max']
    globalAttributes['geospatial_lon_resolution'] = rDS.attrs['geospatial_lon_resolution']   
    globalAttributes['geospatial_vertical_max'] = rDS.attrs['geospatial_vertical_max']
    globalAttributes['geospatial_vertical_resolution'] = rDS.attrs['geospatial_vertical_resolution']     
    globalAttributes['time_coverage_start'] = timeCoverageStart.strftime('%Y-%m-%dT%H:%M:%SZ')
    globalAttributes['time_coverage_end'] = timeCoverageEnd.strftime('%Y-%m-%dT%H:%M:%SZ')
    globalAttributes['time_coverage_resolution'] = rDS.attrs['time_coverage_resolution']
    globalAttributes['time_coverage_duration'] = timeCoverageDuration
    globalAttributes['area'] = rDS.attrs['area']    
    globalAttributes['citation'] += networkData.iloc[0]['citation_statement']
    globalAttributes['processing_level'] = '2B'
    globalAttributes['manufacturer'] = rDS.attrs['manufacturer']
    globalAttributes['sensor_model'] = rDS.attrs['manufacturer']
    
    creationDate = dt.datetime.utcnow()
    globalAttributes['date_created'] = creationDate.strftime('%Y-%m-%dT%H:%M:%SZ')
    globalAttributes['date_modified'] = creationDate.strftime('%Y-%m-%dT%H:%M:%SZ')
    globalAttributes['history'] = 'Data measured from ' + timeCoverageStart.strftime('%Y-%m-%dT%H:%M:%SZ') + ' to ' \
                                + timeCoverageEnd.strftime('%Y-%m-%dT%H:%M:%SZ') + '. netCDF file created at ' \
                                + creationDate.strftime('%Y-%m-%dT%H:%M:%SZ') + ' by the European HFR Node.'        
    
    # Remove spatial_resolution global attribute (only admitted for totals)
    globalAttributes.pop('spatial_resolution')
    
    # Add global attributes to the DataSet
    instacDS.attrs = globalAttributes
        
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
                    instacDS[vv].encoding['_FillValue'] = netCDF4.default_fillvals[np.dtype(dataPacking[vv]['dtype']).kind + str(np.dtype(dataPacking[vv]['dtype']).itemsize)]
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
                instacDS[cc].encoding['_FillValue'] = netCDF4.default_fillvals[np.dtype(dataPacking[cc]['dtype']).kind + str(np.dtype(dataPacking[cc]['dtype']).itemsize)]
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

def buildEHNradialFilename(networkID,siteCode,ts,ext):
    """
    This function builds the filename for radial files according to 
    the structure used by the European HFR Node, i.e. networkID-stationID_YYYY_MM_DD_hhmm.
    
    INPUT:
        networkID: ID of the HFR network
        siteCode: code of the radial site
        ts: timestamp as datetime object
        ext: file extension
        
    OUTPUT:
        radFilename: filename for radial file.
    """
    # Get the time related part of the filename
    timeStr = ts.strftime("_%Y_%m_%d_%H%M")
    
    # Build the filename
    radFilename = networkID + '-' + siteCode + timeStr + ext
    
    return radFilename

def buildEHNradialFolder(basePath,siteCode,ts,vers):
    """
    This function builds the folder structure for storing  radial files according to 
    the structure used by the European HFR Node, i.e. YYYY/YYYY_MM/YYYY_MM_DD/.
    
    INPUT:
        basePath: base path
        siteCode: code of the radial site
        ts: timestamp as datetime object
        vers: version of the data model
        
    OUTPUT:
        radFolder: folder path for storing radial files.
    """
    # Strip trailing slash character
    basePath = basePath.rstrip('/')
    
    # Get the time related part of the path
    timeStr = ts.strftime("/%Y/%Y_%m/%Y_%m_%d/")
    
    # Build the folder path
    radFolder = basePath + '/' + vers + '/' + siteCode + timeStr
    
    return radFolder

def velocityMedianInDistLimits(cell,radData,distLim,g):
    """
    This function evaluates the median of all radial velocities contained in radData
    lying within a distance of distLim km from the origin grid cell.
    The native CRS of the Radial is used for distance calculations.
    
    INPUT:
        cell: Series containing longitude and latitude of the origin grid cell
        radData: DataFrame containing radial data
        distLim: range limit in km for selecting velocities for median calculation
        g: Geod object according to the Radial CRS
        
    OUTPUT:
        median: median of the selected velocities.
    """
    # Convert grid cell Series and radial bins DataFrame to numpy arrays
    cell = cell.to_numpy()
    radLon = radData['LOND'].to_numpy()
    radLat = radData['LATD'].to_numpy() 
    # Evaluate distances between the origin grid cell and radial bins
    az12,az21,cellToRadDist = g.inv(len(radLon)*[cell[0]],len(radLat)*[cell[1]],radLon,radLat)
    
    # Remove the origin from the radial data (i.e. distance = 0)
    radData = radData.drop(radData[cellToRadDist == 0].index)
    # Remove the origin from the distance array (i.e. distance = 0)
    cellToRadDist = cellToRadDist[cellToRadDist != 0]
    
    # Figure out which radial bins are within the range limit from the origin cell
    distSelectionIndices = np.where(cellToRadDist < distLim*1000)[0].tolist()
    
    # Evaluate the median of the selected velocities
    # median = np.median(radData.iloc[distSelectionIndices]['VELO'])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        median = np.nanmedian(radData.iloc[distSelectionIndices]['VELO'])
            
    return median


class Radial(fileParser):
    """
    Radial Subclass.

    This class should be used when loading CODAR (.ruv) or WERA (.crad_ascii) radial files.
    This class utilizes the generic LLUV and CRAD classes.
    """
    def __init__(self, fname, replace_invalid=True, mask_over_land=False, empty_radial=False):
        logging.info('Loading radial file: {}'.format(fname))
        super().__init__(fname)

        if self._iscorrupt:
            return

        self.data = pd.DataFrame()

        for key in self._tables.keys():
            table = self._tables[key]
            if 'LLUV' in table['TableType']:
                self.data = table['data']
            elif 'CRAD' in table['TableType']:
                self.crad_data = table['data']
            elif 'rads' in table['TableType']:
                self.diagnostics_radial = table['data']
                self.diagnostics_radial['datetime'] = self.diagnostics_radial[['TYRS', 'TMON', 'TDAY', 'THRS', 'TMIN', 'TSEC']].apply(lambda s: dt.datetime(*s), axis=1)
            elif 'rcvr' in table['TableType']:
                self.diagnostics_hardware = table['data']
                self.diagnostics_hardware['datetime'] = self.diagnostics_hardware[['TYRS', 'TMON', 'TDAY', 'THRS', 'TMIN', 'TSEC']].apply(lambda s: dt.datetime(*s), axis=1)
            elif 'RINF' in table['TableType']:
                self.range_information = table['data']
            elif 'MRGS' in table['TableType']:
                self.merge_information = table['data']            
                

        if 'Site' in self.metadata.keys():
            self.metadata['Site'] = re.sub(r'[\W_]+', '', self.metadata['Site'])
        

        if not self.data.empty:

            if replace_invalid:
                self.replace_invalid_values()

            if mask_over_land:
                self.mask_over_land()

            if empty_radial:
                self.empty_radial()

    def __repr__(self):
        return "<Radial: {}>".format(self.file_name)

    def empty_radial(self):
        """
        Create an empty Radial object.
        """

        self.file_path = ''
        self.file_name = ''
        self.full_file = ''
        self.metadata = ''
        self._iscorrupt = False
        self.time = []

        for key in self._tables.keys():
            table = self._tables[key]
            self._tables[key]['TableRows'] = '0'
            if 'LLUV' in table['TableType']:
                self.data.drop(self.data.index[:], inplace=True)
                self._tables[key]['data'] = self.data
            elif 'CRAD' in table['TableType']:
                self.crad_data.drop(self.crad_data.index[:], inplace=True)
                self._tables[key]['data'] = self.crad_data
            elif 'rads' in table['TableType']:
                self.diagnostics_radial.drop(self.diagnostics_radial.index[:], inplace=True)
                self._tables[key]['data'] = self.diagnostics_radial
            elif 'rcvr' in table['TableType']:
                self.diagnostics_hardware.drop(self.diagnostics_hardware.index[:], inplace=True)
                self._tables[key]['data'] = self.diagnostics_hardware
            elif 'RINF' in table['TableType']:
                self.range_information.drop(self.range_information.index[:], inplace=True)
                self._tables[key]['data'] = self.range_information

    def mask_over_land(self, subset=True):
        """
        This function masks the radial vectors lying on land.        
        Radial vector coordinates are checked against a reference file containing information 
        about which locations are over land or in an unmeasurable area (for example, behind an 
        island or point of land). 
        The GeoPandas "naturalearth_lowres" is used as reference.        
        The native CRS of the Radial is used for distance calculations.
        If "subset"  option is set to True, the radial vectors lying on land are removed.
        
        INPUT:
            subset: option enabling the removal of radial vectors on land (if set to True)
            
        OUTPUT:
            waterIndex: list containing the indices of radial vectors lying on water.
        """
        # logging.info('Masking radials over land')
        
        # Load the reference file (GeoPandas "naturalearth_lowres")
        land = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        # land = land[land['continent'] == 'North America']

        # Build the GeoDataFrame containing radial points
        geodata = gpd.GeoDataFrame(
            self.data[['LOND', 'LATD']],
            crs=land.crs.srs.upper(),
            geometry=[
                Point(xy) for xy in zip(self.data.LOND.values, self.data.LATD.values)
            ]
        )
        # Join the GeoDataFrame containing radial points with GeoDataFrame containing leasing areas
        geodata = gpd.tools.sjoin(geodata, land, how='left', predicate='intersects')

        # All data in the continent column that lies over water should be nan.
        waterIndex = geodata['continent'].isna()

        if subset:
            # Subset the data to water only
            self.data = self.data.loc[waterIndex].reset_index()
        else:
            return waterIndex
        
    def plot(self, lon_min=None, lon_max=None, lat_min=None, lat_max=None, shade=False, show=True):
        """
        This function plots the current radial velocity field (i.e. VELU and VELV components) on a 
        Cartesian grid. The grid is defined either from the input values or from the Radial object
        metadata. If no input is passed and no metadata related to the bounding box are present, the
        grid is defined from data content (i.e. LOND and LATD values).
        If 'shade' is False (default), a quiver plot with color and magnitude of the vectors proportional to
        current velocity is produced. If 'shade' is True, a quiver plot with uniform vetor lenghts is produced,
        superimposed to a pseudo-color map representing velocity magnitude.
        
        INPUT:
            lon_min: minimum longitude value in decimal degrees (if None it is taken from Total metadata)
            lon_max: maximum longitude value in decimal degrees (if None it is taken from Total metadata)
            lat_min: minimum latitude value in decimal degrees (if None it is taken from Total metadata)
            lat_max: maximum latitude value in decimal degrees (if None it is taken from Total metadata)
            shade: boolean for enabling/disabling shade plot (default False)
            show: boolean for enabling/disabling plot visualization (default True)
            
        OUTPUT:

        """
        # Initialize figure
        fig = plt.figure(figsize=(24,16),tight_layout = {'pad': 0})
        
        # Get the bounding box limits
        if not lon_min:
            if 'BBminLongitude' in self.metadata:
                lon_min = float(self.metadata['BBminLongitude'].split()[0])
            else:
                lon_min = self.data.LOND.min() - 1
                
        if not lon_max:
            if 'BBmaxLongitude' in self.metadata:
                lon_max = float(self.metadata['BBmaxLongitude'].split()[0])
            else:
                lon_max = self.data.LOND.max() + 1
                
        if not lat_min:
            if 'BBminLatitude' in self.metadata:
                lat_min = float(self.metadata['BBminLatitude'].split()[0])
            else:
                lat_min = self.data.LATD.min() - 1
                
        if not lat_max:
            if 'BBmaxLatitude' in self.metadata:
                lat_max = float(self.metadata['BBmaxLatitude'].split()[0])
            else:
                lat_max = self.data.LATD.max() + 1   
                
        # Evaluate longitude and latitude of the center of the map
        lon_C = (lon_max + lon_min) / 2
        lat_C = (lat_max + lat_min) / 2
            
        # Set the background map
        m = Basemap(llcrnrlon=lon_min, llcrnrlat=lat_min, urcrnrlon=lon_max, urcrnrlat=lat_max, lon_0=lon_C, lat_0=lat_C,
                          resolution = 'i', ellps='WGS84', projection='tmerc')        
        m.drawcoastlines()
        # m.fillcontinents(color='#cc9955', lake_color='white')
        m.fillcontinents()
        m.drawparallels(np.arange(lat_min,lat_max))
        m.drawmeridians(np.arange(lon_min,lon_max))
        m.drawmapboundary(fill_color='white')
        
        # m.bluemarble()
        
        # Get station coordinates and codes
        if not self.is_wera:
            siteLon = float(self.metadata['Origin'].split()[1])
            siteLat = float(self.metadata['Origin'].split()[0])
            siteCode = self.metadata['Site']
        else:
            siteLon = dms2dd(list(map(int,self.metadata['Longitude(deg-min-sec)OfTheCenterOfTheReceiveArray'][:-2].split('-'))))
            siteLat = dms2dd(list(map(int,self.metadata['Latitude(deg-min-sec)OfTheCenterOfTheReceiveArray'][:-2].split('-'))))
            if self.metadata['Latitude(deg-min-sec)OfTheCenterOfTheReceiveArray'][-1] == 'S':
                siteLat = -siteLat
            if self.metadata['Longitude(deg-min-sec)OfTheCenterOfTheReceiveArray'][-1] == 'W':
                siteLon = -siteLon
            siteCode = self.metadata['StationName']
        
        # Compute the native map projection coordinates for the stations
        xS, yS = m(siteLon, siteLat)       
        
        # Plot radial stations
        m.plot(xS,yS,'rD')
        plt.text(xS,yS,siteCode,fontdict={'fontsize': 22, 'fontweight' : 'bold'})
        
        # Plot velocity field
        if shade:
            self.to_xarray_multidimensional()
            
            if self.is_wera:
                # Create grid from longitude and latitude
                [longitudes, latitudes] = np.meshgrid(self.xdr['LONGITUDE'].data, self.xdr['LATITUDE'].data)
                
                # Compute the native map projection coordinates for the pseudo-color cells
                X, Y = m(longitudes, latitudes)
            else:
                # Compute the native map projection coordinates for the pseudo-color cells
                X, Y = m(self.xdr['LONGITUDE'].data, self.xdr['LATITUDE'].data)

            # Create velocity variable in the shape of the grid
            V = abs(self.xdr['VELO'][0,0,:,:].to_numpy())
            # V = V[:-1,:-1]            
            
            # Make the pseudo-color plot
            warnings.simplefilter("ignore", category=UserWarning)
            c = m.pcolormesh(X, Y, V, shading='nearest', cmap=plt.cm.jet, vmin=0, vmax=1)            
            
            # Compute the native map projection coordinates for the vectors
            x, y = m(self.data.LOND, self.data.LATD)
            
            # Create the velocity component variables
            if self.is_wera:
                u = self.data.VELU
                v = self.data.VELV
            else:
                u = self.data.VELU / 100        # CODAR velocities are in cm/s
                v = self.data.VELV / 100        # CODAR velocities are in cm/s
            
            # Make the quiver plot
            m.quiver(x, y, u, v, width=0.001, headwidth=4, headlength=4, headaxislength=4)
            
            warnings.simplefilter("default", category=UserWarning)
            
        else:
            # Compute the native map projection coordinates for the vectors
            x, y = m(self.data.LOND, self.data.LATD)
            
            # Create the velocity variables
            if self.is_wera:
                u = self.data.VELU
                v = self.data.VELV
                vel = abs(self.data.VELO)
            else:
                u = self.data.VELU / 100                # CODAR velocities are in cm/s
                v = self.data.VELV / 100                # CODAR velocities are in cm/s
                vel = abs(self.data.VELO) / 100         # CODAR velocities are in cm/s                
            
            # Make the quiver plot
            m.quiver(x, y, u, v, vel, cmap=plt.cm.jet, width=0.001, headwidth=4, headlength=4, headaxislength=4)
            
        # Add colorbar
        cbar = plt.colorbar()
        cbar.set_label('m/s',fontsize='x-large')
        
        # Add title
        plt.title(self.file_name + ' radial velocity field', fontdict={'fontsize': 30, 'fontweight' : 'bold'})
                
        if show:
            plt.show()
        
        return fig

    # def to_xarray(self, range_min=None, range_max=None, enhance=False, dim='tabular'):
    #     """
    #     Adapted from MATLAB code from Mark Otero
    #     http://cordc.ucsd.edu/projects/mapping/documents/HFRNet_Radial_NetCDF.pdf
    #     :param range_min:
    #     :param range_max:
    #     :return:
    #     """
    #     if dim == 'tabular':
    #         self.to_xarray_tabular(enhance)
    #     elif dim == 'multidimensional':
    #         self.to_xarray_multidimensional(range_min, range_max, enhance)
    #     # Clean radial header
    #     # self.clean_header()
    #
    #     # return ds
    
    def to_xarray_multidimensional(self, range_min=None, range_max=None, enhance=False):
        """
        This function creates a dictionary of xarray DataArrays containing the variables
        of the Radial object multidimensionally expanded along the coordinate axes (T,Z,Y,X). 
        The spatial coordinate axes are chosen based on the file type: (RNGE,BEAR) for 
        Codar radials and (LOND,LATD) for WERA radials. The coordinate limits and steps 
        are taken from radial metadata. Only RNGE limits can be specified by the user.
        Some refinements are performed on Codar data in order to comply CF convention
        for positive velocities and to have velocities in m/s.
        The generated dictionary is attached to the Radial object, named as xdr.
        
        INPUT:
            range_min: minimum range value in km (if None it is taken from Radial metadata)
            range_max: maximum range value in km (if None it is taken from Radial metadata)
            
        OUTPUT:
        """
        # Initialize empty dictionary
        xdr = OrderedDict()
        
        # process Codar radial
        if not self.is_wera:
            # Check range limits   
            if range_min is None:
                if 'RangeMin' in self.metadata:
                    range_min = float(self.metadata['RangeMin'].split()[0])
                else:
                    range_min = self.data.RNGE.min()
            if range_max is None:
                if 'RangeMax' in self.metadata:
                    range_max = float(self.metadata['RangeMax'].split()[0])
                else:
                    range_max = self.data.RNGE.max()
            # Get range step
            if 'RangeResolutionKMeters' in self.metadata:
                range_step = float(self.metadata['RangeResolutionKMeters'].split()[0])
            elif 'RangeResolutionMeters' in self.metadata:
                range_step = float(self.metadata['RangeResolutionMeters'].split()[0]) * 0.001
            else:
                range_step = float(1)
            # build range array
            range_dim = np.arange(
                range_min,
                range_max + range_step,
                range_step
            )
            
            # Check that the constraint on maximum number of range cells is satisfied
            range_dim = range_dim[range_dim<=range_max]
            
            # Get bearing step
            if 'AngularResolution' in self.metadata:
                bearing_step = float(self.metadata['AngularResolution'].split()[0])
            else:
                bearing_step = float(1)
            # build bearing array
            # bearing_dim = np.arange(1, 361, 1).astype(np.float)  # Complete 360 degree bearing coordinate allows for better aggregation
            bearing_dim_1 = np.sort(np.arange(np.min(self.data['BEAR']),-bearing_step,-bearing_step))
            bearing_dim_2 = np.sort(np.arange(np.min(self.data['BEAR']),np.max(self.data['BEAR'])+bearing_step,bearing_step))
            bearing_dim_3 = np.sort(np.arange(np.max(self.data['BEAR']),360,bearing_step))
            bearing_dim = np.unique(np.concatenate((bearing_dim_1,bearing_dim_2,bearing_dim_3),axis=None))
            bearing_dim = bearing_dim[(bearing_dim>=0) & (bearing_dim<=360)]
    
            # Create radial grid from bearing and range
            [bearing, ranges] = np.meshgrid(bearing_dim, range_dim)
            
            # Get the ellipsoid of the radial coordinate reference system
            if 'GreatCircle' in self.metadata:
                radEllps = self.metadata['GreatCircle'].split()[0].replace('"','')            
            else:
                radEllps = 'WGS84'
            # Create Geod object with the retrieved ellipsoid
            g = Geod(ellps=radEllps)
    
            # Calculate lat/lons from origin, bearing, and ranges
            latlon = [float(x) for x in re.findall(r"[-+]?\d*\.\d+|\d+", self.metadata['Origin'])]
            bb = bearing.flatten()
            rr = ranges.flatten() * 1000        # distances to be expressed in meters for Geod
            lond, latd, backaz = g.fwd(len(bb)*[latlon[1]], len(bb)*[latlon[0]], bb, rr)
            lond = np.array(lond)
            latd = np.array(latd)
            lond = lond.reshape(bearing.shape)
            latd = latd.reshape(bearing.shape)            
            
            # Find grid indices from radial grid (bearing, ranges)
            range_map_idx = np.tile(np.nan, self.data['RNGE'].shape)
            bearing_map_idx = np.tile(np.nan, self.data['BEAR'].shape)
    
            for i, line in enumerate(self.data['RNGE']):
                range_map_idx[i] = np.argmin(np.abs(range_dim - self.data.RNGE[i]))
                bearing_map_idx[i] = np.argmin(np.abs(bearing_dim - self.data.BEAR[i]))
                
            # Set X and Y coordinate mappings
            X_map_idx = bearing_map_idx         # BEAR is X axis
            Y_map_idx = range_map_idx           # RNGE is Y axis
                
            # Create dictionary containing variables from dataframe in the shape of radial grid
            d = {key: np.tile(np.nan, bearing.shape) for key in self.data.keys()}
        
        # process WERA radials
        else:
            # Get longitude limits and step
            if 'TopLeftLongitude' in self.metadata:
                topLeftLon = float(self.metadata['TopLeftLongitude'].split()[0])
            else:
                topLeftLon = float(0)
            if 'nx' in self.metadata:
                cellsLon = int(self.metadata['nx'].split()[0])
            else:
                cellsLon = 100
                
            # Get latitude limits and step
            if 'TopLeftLatitude' in self.metadata:
                topLeftLat = float(self.metadata['TopLeftLatitude'].split()[0])
            else:
                topLeftLat = float(90)
            if 'ny' in self.metadata:
                cellsLat = int(self.metadata['ny'].split()[0])
            else:
                cellsLat = 100
            
            # Get cell size in km
            if 'DGT' in self.metadata:
                cellSize = float(self.metadata['DGT'].split()[0])
            else:
                cellSize = float(2)
            
            # Generate grid coordinates
            gridGS = createLonLatGridFromTopLeftPointWera(topLeftLon, topLeftLat, cellSize, cellsLon, cellsLat)
            # extract longitudes and latitude from grid GeoSeries and insert them into numpy arrays
            lon_dim = np.unique(gridGS.x.to_numpy())
            lat_dim = np.unique(gridGS.y.to_numpy())
            # manage antimeridian crossing
            lon_dim = np.concatenate((lon_dim[lon_dim>=0],lon_dim[lon_dim<0]))
            
            # # Get the longitude and latitude values of the radial measurements
            # unqLon = np.sort(np.unique(self.data['LOND']))
            # unqLat = np.sort(np.unique(self.data['LATD']))
            
            # # Insert unqLon and unqLat values to replace the closest in lon_dim and lat_dim 
            # replaceIndLon = abs(unqLon[None, :] - lon_dim[:, None]).argmin(axis=0).tolist()
            # replaceIndLat = abs(unqLat[None, :] - lat_dim[:, None]).argmin(axis=0).tolist()
            # lon_dim[replaceIndLon] = unqLon
            # lat_dim[replaceIndLat] = unqLat            
    
            # Create radial grid from longitude and latitude
            [longitudes, latitudes] = np.meshgrid(lon_dim, lat_dim)
    
            # Find grid indices from lon/lat grid (longitudes, latitudes)
            lat_map_idx = np.tile(np.nan, self.data['LATD'].shape)
            lon_map_idx = np.tile(np.nan, self.data['LOND'].shape)
    
            for i, line in enumerate(self.data['LATD']):
                lat_map_idx[i] = np.argmin(np.abs(lat_dim - self.data.LATD[i]))
                lon_map_idx[i] = np.argmin(np.abs(lon_dim - self.data.LOND[i]))
                
            # set X and Y coordinate mappings
            X_map_idx = lon_map_idx             # LONGITUDE is X axis
            Y_map_idx = lat_map_idx             # LATITUDE is Y axis
                
            # create dictionary containing variables from dataframe in the shape of radial grid
            d = {key: np.tile(np.nan, longitudes.shape) for key in self.data.keys()}        
            
        # Remap all variables
        for k, v in d.items():
            v[Y_map_idx.astype(int), X_map_idx.astype(int)] = self.data[k]
            d[k] = v

        # Add extra dimensions for time (T) and depth (Z) - CF Standard: T, Z, Y, X -> T=axis0, Z=axis1
        d = {k: np.expand_dims(np.float32(v), axis=(0,1)) for (k, v) in d.items()}            

        # Drop LOND, LATD, BEAR and RNGE variables (they are set as coordinates of the DataSet)
        if self.is_wera:
            d.pop('LOND')
            d.pop('LATD') 
        else:
            d.pop('BEAR')
            d.pop('RNGE')
            d.pop('LOND')
            d.pop('LATD') 

        # Refine Codar data
        if not self.is_wera:
            # Flip sign so positive velocities are away from the radar as per CF conventions (only for Codar radials)
            flips = ['MINV', 'MAXV', 'VELO']
            for f in flips:
                if f in d:
                    d[f] = -d[f]
            # Scale velocities to be in m/s (only for Codar radials)
            toMs = ['VELU', 'VELV', 'VELO', 'ESPC', 'ETMP', 'MINV', 'MAXV']
            for t in toMs:
                if t in d:
                    d[t] = d[t]*0.01

        # Evaluate timestamp as number of days since 1950-01-01T00:00:00Z
        timeDelta = self.time - dt.datetime.strptime('1950-01-01T00:00:00Z','%Y-%m-%dT%H:%M:%SZ')
        ncTime = timeDelta.days + timeDelta.seconds / (60*60*24)
        
        # Add all variables as xarray
        if not self.is_wera:
            for k, v in d.items():
                xdr[k] = xr.DataArray(v,
                                         dims={'TIME': v.shape[0], 'DEPTH': v.shape[1], 'RNGE': v.shape[2], 'BEAR': v.shape[3]},
                                         coords={'TIME': [ncTime],
                                                 'DEPTH': [0],
                                                 'RNGE': range_dim,
                                                 'BEAR': bearing_dim})
        else:
            for k, v in d.items():
                xdr[k] = xr.DataArray(v,
                                         dims={'TIME': v.shape[0], 'DEPTH': v.shape[1], 'LATITUDE': v.shape[2], 'LONGITUDE': v.shape[3]},
                                         coords={'TIME': [ncTime],
                                                 'DEPTH': [0],
                                                 'LATITUDE': lat_dim,
                                                 'LONGITUDE': lon_dim})
            
        # Add longitudes and latitudes evaluated from bearing/range grid to the dictionary (BEAR and RNGE are coordinates) for Codar data
        if not self.is_wera:
            xdr['LONGITUDE'] = xr.DataArray(np.float32(lond),
                                         dims={'RNGE': lond.shape[0], 'BEAR': lond.shape[1]},
                                         coords={'RNGE': range_dim,
                                                 'BEAR': bearing_dim})
            xdr['LATITUDE'] = xr.DataArray(np.float32(latd),
                                         dims={'RNGE': latd.shape[0], 'BEAR': latd.shape[1]},
                                         coords={'RNGE': range_dim,
                                                 'BEAR': bearing_dim})          
            
        # Add DataArray for coordinate variables
        xdr['TIME'] = xr.DataArray(ncTime,
                                 dims={'TIME': len(pd.date_range(self.time, periods=1))},
                                 coords={'TIME': [ncTime]})
        xdr['DEPTH'] = xr.DataArray(0,
                                 dims={'DEPTH': 1},
                                 coords={'DEPTH': [0]})
        if not self.is_wera:
            xdr['RNGE'] = xr.DataArray(range_dim,
                                 dims={'RNGE': range_dim},
                                 coords={'RNGE': range_dim})
            xdr['BEAR'] = xr.DataArray(bearing_dim,
                                     dims={'BEAR': bearing_dim},
                                     coords={'BEAR': bearing_dim})
        else:
            xdr['LATITUDE'] = xr.DataArray(lat_dim,
                                 dims={'LATITUDE': lat_dim},
                                 coords={'LATITUDE': lat_dim})
            xdr['LONGITUDE'] = xr.DataArray(lon_dim,
                                     dims={'LONGITUDE': lon_dim},
                                     coords={'LONGITUDE': lon_dim})              
        
        # Attach the dictionary to the Radial object
        self.xdr = xdr
        
        return
    
    def check_ehn_mandatory_variables(self):
        """
        This function checks if the Radial object contains all the mandatory data variables
        (i.e. not coordinate variables) required by the European standard data model developed in the framework of the 
        EuroGOOS HFR Task Team.
        Missing variables are appended to the DataFrame containing data, filled with NaNs.
        
        INPUT:            
            
        OUTPUT:
        """
        # Set mandatory variables based on the HFR manufacturer
        if self.is_wera:
            chkVars = ['VELU', 'VELV', 'VELO', 'HEAD', 'HCSS', 'EACC']
        else:
            chkVars = ['VELU', 'VELV', 'ESPC', 'ETMP', 'MAXV', 'MINV', 'ERSC', 'ERTC', 'XDST', 'YDST', 'VELO', 'HEAD', 'SPRC']
            
        # Check variables and add missing ones
        for vv in chkVars:
            if vv not in self.data.columns:
                self.data[vv] = np.nan
        
        return
    
    
    def apply_ehn_datamodel(self, network_data, station_data, version):
        """
        This function applies the European standard data model developed in the
        framework of the EuroGOOS HFR Task Team to the Radial object.
        The Radial object content is stored into an xarray Dataset built from the
        xarray DataArrays created by the Radial method to_xarray_multidimensional.
        Variable data types and data packing information are collected from
        "Data_Models/EHN/Radials/Radial_Data_Packing.json" file.
        Variable attribute schema is collected from 
        "Data_Models/EHN/Radials/Radial_Variables.json" file.
        Global attribute schema is collected from 
        "Data_Models/EHN/Global_Attributes.json" file.
        Global attributes are created starting from Radial object metadata and from 
        DataFrames containing the information about HFR network and radial station
        read from the EU HFR NODE database.
        The generated xarray Dataset is attached to the Radial object, named as xds.
        
        INPUT:
            network_data: DataFrame containing the information of the network to which the radial site belongs
            station_data: DataFrame containing the information of the radial site that produced the radial
            version: version of the data model
            
            
        OUTPUT:
        """
        # Set the netCDF format
        ncFormat = 'NETCDF4_CLASSIC'
        
        # Expand Radial object variables along the coordinate axes
        self.to_xarray_multidimensional()
        
        # Set auxiliary coordinate sizes
        maxsiteSize = 1
        refmaxSize = 10
        maxinstSize = 1
        
        # Get data packing information per variable
        f = open('Data_Models/EHN/Radials/Radial_Data_Packing.json')
        dataPacking = json.loads(f.read())
        f.close()
        
        # Get variable attributes
        f = open('Data_Models/EHN/Radials/Radial_Variables.json')
        radVariables = json.loads(f.read())
        f.close()
        
        # Get global attributes
        f = open('Data_Models/EHN/Global_Attributes.json')
        globalAttributes = json.loads(f.read())
        f.close()
        
        # Rename velocity related variables
        self.xdr['DRVA'] = self.xdr.pop('HEAD')
        self.xdr['RDVA'] = self.xdr.pop('VELO')
        self.xdr['EWCT'] = self.xdr.pop('VELU')
        self.xdr['NSCT'] = self.xdr.pop('VELV')
        
        # Drop unnecessary DataArrays from the DataSet
        if not self.is_wera:
            self.xdr.pop('VFLG')
        toDrop = []
        for vv in self.xdr:
            if vv not in radVariables.keys():
                toDrop.append(vv)
        for rv in toDrop:
            self.xdr.pop(rv)
            
        # Add coordinate reference system to the dictionary
        self.xdr['crs'] = xr.DataArray(int(0), )  
        
        # Add antenna related variables to the dictionary
        # Number of antennas
        self.xdr['NARX'] = xr.DataArray([np.expand_dims(station_data.iloc[0]['number_of_receive_antennas'],axis=0)], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xdr['NATX'] = xr.DataArray([np.expand_dims(station_data.iloc[0]['number_of_transmit_antennas'],axis=0)], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        
        # Longitude and latitude of antennas
        self.xdr['SLTR'] = xr.DataArray([np.expand_dims(station_data.iloc[0]['site_lat'],axis=0)], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xdr['SLNR'] = xr.DataArray([np.expand_dims(station_data.iloc[0]['site_lon'],axis=0)], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xdr['SLTT'] = xr.DataArray([np.expand_dims(station_data.iloc[0]['site_lat'],axis=0)], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xdr['SLNT'] = xr.DataArray([np.expand_dims(station_data.iloc[0]['site_lon'],axis=0)], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        
        # Codes of antennas
        antCode = ('%s' % station_data.iloc[0]['station_id']).encode()
        self.xdr['SCDR'] = xr.DataArray(np.array([[antCode]]), dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xdr['SCDR'].encoding['char_dim_name'] = 'STRING' + str(len(antCode))
        self.xdr['SCDT'] = xr.DataArray(np.array([[antCode]]), dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xdr['SCDT'].encoding['char_dim_name'] = 'STRING' + str(len(antCode))
                
        # Add SDN namespace variables to the dictionary
        siteCode = ('%s' % station_data.iloc[0]['network_id']).encode()
        self.xdr['SDN_CRUISE'] = xr.DataArray([siteCode], dims={'TIME': len(pd.date_range(self.time, periods=1))})
        self.xdr['SDN_CRUISE'].encoding['char_dim_name'] = 'STRING' + str(len(siteCode))
        platformCode = ('%s' % station_data.iloc[0]['network_id'] + '-' + station_data.iloc[0]['station_id']).encode()
        self.xdr['SDN_STATION'] = xr.DataArray([platformCode], dims={'TIME': len(pd.date_range(self.time, periods=1))})
        self.xdr['SDN_STATION'].encoding['char_dim_name'] = 'STRING' + str(len(platformCode))
        ID = ('%s' % platformCode.decode() + '_' + self.time.strftime('%Y-%m-%dT%H:%M:%SZ')).encode()
        self.xdr['SDN_LOCAL_CDI_ID'] = xr.DataArray([ID], dims={'TIME': len(pd.date_range(self.time, periods=1))})
        self.xdr['SDN_LOCAL_CDI_ID'].encoding['char_dim_name'] = 'STRING' + str(len(ID))
        self.xdr['SDN_EDMO_CODE'] = xr.DataArray([np.expand_dims(station_data.iloc[0]['EDMO_code'],axis=0)], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXINST': maxinstSize})
        sdnRef = ('%s' % network_data.iloc[0]['metadata_page']).encode()
        self.xdr['SDN_REFERENCES'] = xr.DataArray([sdnRef], dims={'TIME': len(pd.date_range(self.time, periods=1))})
        self.xdr['SDN_REFERENCES'].encoding['char_dim_name'] = 'STRING' + str(len(sdnRef))
        sdnXlink = ('%s' % '<sdn_reference xlink:href=\"' + sdnRef.decode() + '\" xlink:role=\"\" xlink:type=\"URL\"/>').encode()
        self.xdr['SDN_XLINK'] = xr.DataArray(np.array([[sdnXlink]]), dims={'TIME': len(pd.date_range(self.time, periods=1)), 'REFMAX': refmaxSize})
        self.xdr['SDN_XLINK'].encoding['char_dim_name'] = 'STRING' + str(len(sdnXlink))
        
        # Add spatial and temporal coordinate QC variables (set to good data due to the nature of HFR system)
        self.xdr['TIME_QC'] = xr.DataArray([1],dims={'TIME': len(pd.date_range(self.time, periods=1))})
        self.xdr['POSITION_QC'] = self.xdr['QCflag'] * 0 + 1
        self.xdr['DEPTH_QC'] = xr.DataArray([1],dims={'TIME': len(pd.date_range(self.time, periods=1))})
            
        # Create DataSet from DataArrays
        self.xds = xr.Dataset(self.xdr)
        
        # Add data variable attributes to the DataSet
        for vv in self.xds:
            self.xds[vv].attrs = radVariables[vv]
            
        # Update QC variable attribute "comment" for inserting test thresholds and attribute "flag_values" for assigning the right data type
        for qcv in self.metadata['QCTest']:
            if qcv in self.xds:
                self.xds[qcv].attrs['comment'] = self.xds[qcv].attrs['comment'] + ' ' + self.metadata['QCTest'][qcv]
                self.xds[qcv].attrs['flag_values'] = list(np.int_(self.xds[qcv].attrs['flag_values']).astype(dataPacking[qcv]['dtype']))
        for qcv in ['TIME_QC', 'POSITION_QC', 'DEPTH_QC']:
            if qcv in self.xds:
                self.xds[qcv].attrs['flag_values'] = list(np.int_(self.xds[qcv].attrs['flag_values']).astype(dataPacking[qcv]['dtype']))
                
        # Add coordinate variable attributes to the DataSet
        for cc in self.xds.coords:
            self.xds[cc].attrs = radVariables[cc]
        
        # Remove axis attributes from LONGITUDE and LATITUDE variables for polar geometry radials (i.e. Codar radials)
        if not self.is_wera:
            self.xds.LONGITUDE.attrs.pop('axis')
            self.xds.LATITUDE.attrs.pop('axis')
            
        # Evaluate measurement maximum depth
        vertMax = 3e8 / (8*np.pi * station_data.iloc[0]['transmit_central_frequency']*1e6)
        
        # Evaluate time coverage start, end, resolution and duration
        timeCoverageStart = self.time - relativedelta(minutes=station_data.iloc[0]['temporal_resolution']/2)
        timeCoverageEnd = self.time + relativedelta(minutes=station_data.iloc[0]['temporal_resolution']/2)
        timeResRD = relativedelta(minutes=station_data.iloc[0]['temporal_resolution'])
        timeCoverageResolution = 'PT'
        if timeResRD.hours !=0:
            timeCoverageResolution += str(int(timeResRD.hours)) + 'H'
        if timeResRD.minutes !=0:
            timeCoverageResolution += str(int(timeResRD.minutes)) + 'M'
        if timeResRD.seconds !=0:
            timeCoverageResolution += str(int(timeResRD.seconds)) + 'S'        
            
        # Fill global attributes
        globalAttributes['site_code'] = siteCode.decode()
        globalAttributes['platform_code'] = platformCode.decode()
        if station_data.iloc[0]['oceanops_ref'] != None:
            globalAttributes['oceanops_ref'] = station_data.iloc[0]['oceanops_ref']
            globalAttributes['wmo_platform_code'] = station_data.iloc[0]['wmo_code']
            globalAttributes['wigos_id'] = station_data.iloc[0]['wigos_id']
        globalAttributes['doa_estimation_method'] = station_data.iloc[0]['DoA_estimation_method']
        globalAttributes['calibration_type'] = station_data.iloc[0]['calibration_type']
        globalAttributes['last_calibration_date'] = station_data.iloc[0]['last_calibration_date'].strftime('%Y-%m-%dT%H:%M:%SZ')
        globalAttributes['calibration_link'] = station_data.iloc[0]['calibration_link']
        # globalAttributes['title'] = network_data.iloc[0]['title']
        globalAttributes['title'] = 'Near Real Time Surface Ocean Radial Velocity by ' + globalAttributes['platform_code']
        globalAttributes['summary'] = station_data.iloc[0]['summary']
        globalAttributes['institution'] = station_data.iloc[0]['institution_name']
        globalAttributes['institution_edmo_code'] = str(station_data.iloc[0]['EDMO_code'])
        globalAttributes['institution_references'] = station_data.iloc[0]['institution_website']
        globalAttributes['id'] = ID.decode()
        globalAttributes['project'] = network_data.iloc[0]['project']
        globalAttributes['comment'] = network_data.iloc[0]['comment']
        globalAttributes['network'] = network_data.iloc[0]['network_name']
        globalAttributes['data_type'] = globalAttributes['data_type'].replace('current data', 'radial current data')
        globalAttributes['geospatial_lat_min'] = str(network_data.iloc[0]['geospatial_lat_min'])
        globalAttributes['geospatial_lat_max'] = str(network_data.iloc[0]['geospatial_lat_max'])
        if not self.is_wera:
            globalAttributes['geospatial_lat_resolution'] = str(np.abs(np.mean(np.diff(self.xds.LATITUDE.values[:,0]))))     
        else:
            globalAttributes['geospatial_lat_resolution'] = self.metadata['DGT']
        globalAttributes['geospatial_lon_min'] = str(network_data.iloc[0]['geospatial_lon_min'])
        globalAttributes['geospatial_lon_max'] = str(network_data.iloc[0]['geospatial_lon_max'])
        if not self.is_wera:
            globalAttributes['geospatial_lon_resolution'] = str(np.abs(np.mean(np.diff(self.xds.LONGITUDE.values[:,0]))))     
        else:
            globalAttributes['geospatial_lon_resolution'] = self.metadata['DGT']
        globalAttributes['geospatial_vertical_max'] = str(vertMax)
        globalAttributes['geospatial_vertical_resolution'] = str(vertMax)        
        globalAttributes['time_coverage_start'] = timeCoverageStart.strftime('%Y-%m-%dT%H:%M:%SZ')
        globalAttributes['time_coverage_end'] = timeCoverageEnd.strftime('%Y-%m-%dT%H:%M:%SZ')
        globalAttributes['time_coverage_resolution'] = timeCoverageResolution
        globalAttributes['time_coverage_duration'] = timeCoverageResolution
        globalAttributes['area'] = network_data.iloc[0]['area']
        globalAttributes['format_version'] = version
        globalAttributes['netcdf_format'] = ncFormat
        globalAttributes['citation'] += network_data.iloc[0]['citation_statement']
        globalAttributes['license'] = network_data.iloc[0]['license']
        globalAttributes['acknowledgment'] = network_data.iloc[0]['acknowledgment']
        globalAttributes['processing_level'] = '2B'
        globalAttributes['contributor_name'] = network_data.iloc[0]['contributor_name']
        globalAttributes['contributor_role'] = network_data.iloc[0]['contributor_role']
        globalAttributes['contributor_email'] = network_data.iloc[0]['contributor_email']
        globalAttributes['manufacturer'] = station_data.iloc[0]['manufacturer']
        globalAttributes['sensor_model'] = station_data.iloc[0]['manufacturer']
        globalAttributes['software_version'] = version
        
        creationDate = dt.datetime.utcnow()
        globalAttributes['metadata_date_stamp'] = creationDate.strftime('%Y-%m-%dT%H:%M:%SZ')
        globalAttributes['date_created'] = creationDate.strftime('%Y-%m-%dT%H:%M:%SZ')
        globalAttributes['date_modified'] = creationDate.strftime('%Y-%m-%dT%H:%M:%SZ')
        globalAttributes['history'] = 'Data measured at ' + self.time.strftime('%Y-%m-%dT%H:%M:%SZ') + '. netCDF file created at ' \
                                    + creationDate.strftime('%Y-%m-%dT%H:%M:%SZ') + ' by the European HFR Node.'        
        
        # Add global attributes to the DataSet
        self.xds.attrs = globalAttributes
            
        # Encode data types, data packing and _FillValue for the data variables of the DataSet
        for vv in self.xds:
            if vv in dataPacking:
                if 'dtype' in dataPacking[vv]:
                    self.xds[vv].encoding['dtype'] = dataPacking[vv]['dtype']
                if 'scale_factor' in dataPacking[vv]:
                    self.xds[vv].encoding['scale_factor'] = dataPacking[vv]['scale_factor']                
                if 'add_offset' in dataPacking[vv]:
                    self.xds[vv].encoding['add_offset'] = dataPacking[vv]['add_offset']
                if 'fill_value' in dataPacking[vv]:
                    self.xds[vv].encoding['_FillValue'] = netCDF4.default_fillvals[np.dtype(dataPacking[vv]['dtype']).kind + str(np.dtype(dataPacking[vv]['dtype']).itemsize)]
                else:
                    self.xds[vv].encoding['_FillValue'] = None
                    
        # Update valid_min and valid_max variable attributes according to data packing
        for vv in self.xds:
            if 'valid_min' in radVariables[vv]:
                if ('scale_factor' in dataPacking[vv]) and ('add_offset' in dataPacking[vv]):
                    self.xds[vv].attrs['valid_min'] = np.float_(((radVariables[vv]['valid_min'] - dataPacking[vv]['add_offset']) / dataPacking[vv]['scale_factor'])).astype(dataPacking[vv]['dtype'])
                else:
                    self.xds[vv].attrs['valid_min'] = np.float_(radVariables[vv]['valid_min']).astype(dataPacking[vv]['dtype'])
            if 'valid_max' in radVariables[vv]:             
                if ('scale_factor' in dataPacking[vv]) and ('add_offset' in dataPacking[vv]):
                    self.xds[vv].attrs['valid_max'] = np.float_(((radVariables[vv]['valid_max'] - dataPacking[vv]['add_offset']) / dataPacking[vv]['scale_factor'])).astype(dataPacking[vv]['dtype'])
                else:
                    self.xds[vv].attrs['valid_max'] = np.float_(radVariables[vv]['valid_max']).astype(dataPacking[vv]['dtype'])
            
        # Encode data types and avoid data packing, valid_min, valid_max and _FillValue for the coordinate variables of the DataSet
        for cc in self.xds.coords:
            if cc in dataPacking:
                if 'dtype' in dataPacking[cc]:
                    self.xds[cc].encoding['dtype'] = dataPacking[cc]['dtype']
                if 'valid_min' in radVariables[cc]:
                    del self.xds[cc].attrs['valid_min']
                if 'valid_max' in radVariables[cc]:
                    del self.xds[cc].attrs['valid_max']
                self.xds[cc].encoding['_FillValue'] = None
               
        return

    # def to_xarray_tabular(self, range_min=None, range_max=None, enhance=False):
    #     """
    #     :param range_min:
    #     :param range_max:
    #     :return:
    #     """
    #     logging.info('Converting radial matrix to tabular dataset')

    #     # Clean radial header
    #     # self.clean_header()

    #     # get timestamp from radial metadata
    #     timestamp = dt.datetime(*[int(s) for s in self.metadata['TimeStamp'].split()])

    #     self.data['time'] = timestamp
    #     self.data.set_index('time', inplace=True)

    #     # Intitialize xarray dataset
    #     ds = self.data.to_xarray()
    #     # ds.coords['time'] = pd.date_range(timestamp, periods=1)
    #     # ds.expand_dims('time').assign_coords(time=('time', [timestamp]))

    #     # Check if calculated longitudes and latitudes align with given longitudes and latitudes
    #     # plt.plot(ds.lon, ds.lat, 'bo', ds.LOND.squeeze(), ds.LATD.squeeze(), 'rx')

    #     # Flip sign so positive velocities are away from the radar as per cf conventions
    #     flips = ['MINV', 'MAXV', 'VELO']
    #     for f in flips:
    #         if f in ds:
    #             ds[f] = -ds[f]

    #     return ds

    # def enhance_xarray(self, xds):
    #     rename = dict(
    #         VELU='u',
    #         VELV='v',
    #         VFLG='vector_flag',
    #         ESPC='spatial_quality',
    #         ETMP='temporal_quality',
    #         MAXV='velocity_max',
    #         MINV='velocity_min',
    #         ERSC='spatial_count',
    #         ERTC='temporal_count',
    #         XDST='dist_east_from_origin',
    #         YDST='dist_north_from_origin',
    #         VELO='velocity',
    #         HEAD='heading',
    #         SPRC='range_cell',
    #         EACC='accuracy',  # WERA specific
    #         LOND='lon',
    #         LATD='lat',
    #         BEAR='bearing',
    #         RNGE='range'
    #     )

    #     rename_qc = dict()

    #     # rename variables to something meaningful if they existin
    #     # in the xarray dataset
    #     existing_renames = { k: v for k, v in rename.items() if k in xds }
    #     xds = xds.rename(existing_renames)

    #     # set time attribute
    #     xds['time'].attrs['standard_name'] = 'time'

    #     # Set lon attributes
    #     xds['lon'].attrs['long_name'] = 'Longitude'
    #     xds['lon'].attrs['standard_name'] = 'longitude'
    #     xds['lon'].attrs['short_name'] = 'lon'
    #     xds['lon'].attrs['units'] = 'degrees_east'
    #     xds['lon'].attrs['axis'] = 'X'
    #     xds['lon'].attrs['valid_min'] = np.float32(-180.0)
    #     xds['lon'].attrs['valid_max'] = np.float32(180.0)

    #     # Set lat attributes
    #     xds['lat'].attrs['long_name'] = 'Latitude'
    #     xds['lat'].attrs['standard_name'] = 'latitude'
    #     xds['lat'].attrs['short_name'] = 'lat'
    #     xds['lat'].attrs['units'] = 'degrees_north'
    #     xds['lat'].attrs['axis'] = 'Y'
    #     xds['lat'].attrs['valid_min'] = np.float32(-90.0)
    #     xds['lat'].attrs['valid_max'] = np.float32(90.0)

    #     # Set u attributes
    #     xds['u'].attrs['long_name'] = 'Eastward Surface Current (cm/s)'
    #     xds['u'].attrs['standard_name'] = 'surface_eastward_sea_water_velocity'
    #     xds['u'].attrs['short_name'] = 'u'
    #     xds['u'].attrs['units'] = 'cm s-1'
    #     xds['u'].attrs['valid_min'] = np.float32(-300)
    #     xds['u'].attrs['valid_max'] = np.float32(300)
    #     xds['u'].attrs['coordinates'] = 'lon lat'
    #     xds['u'].attrs['grid_mapping'] = 'crs'

    #     # Set v attributes
    #     xds['v'].attrs['long_name'] = 'Northward Surface Current (cm/s)'
    #     xds['v'].attrs['standard_name'] = 'surface_northward_sea_water_velocity'
    #     xds['v'].attrs['short_name'] = 'v'
    #     xds['v'].attrs['units'] = 'cm s-1'
    #     xds['v'].attrs['valid_min'] = np.float32(-300)
    #     xds['v'].attrs['valid_max'] = np.float32(300)
    #     xds['v'].attrs['coordinates'] = 'lon lat'
    #     xds['v'].attrs['grid_mapping'] = 'crs'

    #     # Set bearing attributes
    #     xds['bearing'].attrs['long_name'] = 'Bearing from origin (away from instrument)'
    #     xds['bearing'].attrs['short_name'] = 'bearing'
    #     xds['bearing'].attrs['units'] = 'degrees'
    #     xds['bearing'].attrs['valid_min'] = np.float32(0)
    #     xds['bearing'].attrs['valid_max'] = np.float32(360)
    #     xds['bearing'].attrs['grid_mapping'] = 'crs'
    #     xds['bearing'].attrs['axis'] = 'Y'

    #     # Set range attributes
    #     xds['range'].attrs['long_name'] = 'Range from origin (away from instrument)'
    #     xds['range'].attrs['short_name'] = 'range'
    #     xds['range'].attrs['units'] = 'km'
    #     xds['range'].attrs['valid_min'] = np.float32(0)
    #     xds['range'].attrs['valid_max'] = np.float32(1000)
    #     xds['range'].attrs['grid_mapping'] = 'crs'
    #     xds['range'].attrs['axis'] = 'X'

    #     # velocity
    #     xds['velocity'].attrs['valid_range'] = [-1000, 1000]
    #     xds['velocity'].attrs['standard_name'] = 'radial_sea_water_velocity_away_from_instrument'
    #     xds['velocity'].attrs['units'] = 'cm s-1'
    #     xds['velocity'].attrs['coordinates'] = 'lon lat'
    #     xds['velocity'].attrs['grid_mapping'] = 'crs'

    #     # heading
    #     if 'heading' in xds:
    #         xds['heading'].attrs['valid_range'] = [0, 3600]
    #         xds['heading'].attrs['standard_name'] = 'direction_of_radial_vector_away_from_instrument'
    #         xds['heading'].attrs['units'] = 'degrees'
    #         xds['heading'].attrs['coordinates'] = 'lon lat'
    #         xds['heading'].attrs['scale_factor'] = 0.1
    #         xds['heading'].attrs['grid_mapping'] = 'crs'

    #     # vector_flag
    #     if 'vector_flag' in xds:
    #         xds['vector_flag'].attrs['long_name'] = 'Vector Flag Masks'
    #         xds['vector_flag'].attrs['valid_range'] = [0, 2048]
    #         xds['vector_flag'].attrs['flag_masks'] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    #         xds['vector_flag'].attrs['flag_meanings'] = 'grid_point_deleted grid_point_near_coast point_measurement no_radial_solution baseline_interpolation exceeds_max_speed invalid_solution solution_beyond_valid_spatial_domain insufficient_angular_resolution reserved reserved'
    #         xds['vector_flag'].attrs['coordinates'] = 'lon lat'
    #         xds['vector_flag'].attrs['grid_mapping'] = 'crs'

    #     # spatial_quality
    #     if 'spatial_quality' in xds:
    #         xds['spatial_quality'].attrs['long_name'] = 'Spatial Quality of radial sea water velocity'
    #         xds['spatial_quality'].attrs['units'] = 'cm s-1'
    #         xds['spatial_quality'].attrs['coordinates'] = 'lon lat'
    #         xds['spatial_quality'].attrs['grid_mapping'] = 'crs'

    #     # temporal_quality
    #     if 'temporal_quality' in xds:
    #         xds['temporal_quality'].attrs['long_name'] = 'Temporal Quality of radial sea water velocity'
    #         xds['temporal_quality'].attrs['units'] = 'cm s-1'
    #         xds['temporal_quality'].attrs['coordinates'] = 'lon lat'
    #         xds['temporal_quality'].attrs['grid_mapping'] = 'crs'

    #     # velocity_max
    #     if 'velocity_max' in xds:
    #         xds['velocity_max'].attrs['long_name'] = 'Maximum Velocity of sea water (away from instrument)'
    #         xds['velocity_max'].attrs['units'] = 'cm s-1'
    #         xds['velocity_max'].attrs['coordinates'] = 'lon lat'
    #         xds['velocity_max'].attrs['grid_mapping'] = 'crs'

    #     # velocity_min
    #     if 'velocity_min' in xds:
    #         xds['velocity_min'].attrs['long_name'] = 'Minimum Velocity of sea water (away from instrument)'
    #         xds['velocity_min'].attrs['units'] = 'cm s-1'
    #         xds['velocity_min'].attrs['coordinates'] = 'lon lat'
    #         xds['velocity_min'].attrs['grid_mapping'] = 'crs'

    #     # spatial_count
    #     if 'spatial_count' in xds:
    #         xds['spatial_count'].attrs['long_name'] = 'Spatial count of sea water velocity (away from instrument)'
    #         xds['spatial_count'].attrs['coordinates'] = 'lon lat'
    #         xds['spatial_count'].attrs['grid_mapping'] = 'crs'

    #     # temporal_count
    #     if 'temporal_count' in xds:
    #         xds['temporal_count'].attrs['long_name'] = 'Temporal count of sea water velocity (away from instrument)'
    #         xds['temporal_count'].attrs['coordinates'] = 'lon lat'
    #         xds['temporal_count'].attrs['grid_mapping'] = 'crs'

    #     # east_dist_from_origin
    #     if 'dist_east_from_origin' in xds:
    #         xds['dist_east_from_origin'].attrs['long_name'] = 'Eastward distance from instrument'
    #         xds['dist_east_from_origin'].attrs['units'] = 'km'
    #         xds['dist_east_from_origin'].attrs['coordinates'] = 'lon lat'
    #         xds['dist_east_from_origin'].attrs['grid_mapping'] = 'crs'

    #     # north_dist_from_origin
    #     if 'dist_north_from_origin' in xds:
    #         xds['dist_north_from_origin'].attrs['long_name'] = 'Northward distance from instrument'
    #         xds['dist_north_from_origin'].attrs['units'] = 'km'
    #         xds['dist_north_from_origin'].attrs['coordinates'] = 'lon lat'
    #         xds['dist_north_from_origin'].attrs['grid_mapping'] = 'crs'

    #     # range_cell
    #     if 'range_cell' in xds:
    #         xds['range_cell'].attrs['long_name'] = 'Cross Spectra Range Cell  of sea water velocity (away from instrument)'
    #         xds['range_cell'].attrs['coordinates'] = 'lon lat'
    #         xds['range_cell'].attrs['grid_mapping'] = 'crs'

    #     # range_cell
    #     if 'accuracy' in xds:
    #         xds['accuracy'].attrs['long_name'] = 'Accuracy'
    #         xds['accuracy'].attrs['coordinates'] = 'lon lat'
    #         xds['accuracy'].attrs['grid_mapping'] = 'crs'
    #         xds['accuracy'].attrs['units'] = 'cm s-1'


    #     # QC06
    #     if 'QC06' in xds:
    #         xds['QC06'].attrs['long_name'] = 'Syntax (QARTOD Test 06) Flag Masks'
    #         xds['QC06'].attrs['valid_range'] = [1, 9]
    #         xds['QC06'].attrs['flag_values'] =  [1, 2, 3, 4, 5]
    #         xds['QC06'].attrs['flag_meanings'] = 'pass not_evaluated suspect fail missing_data'
    #         xds['QC06'].attrs['coordinates'] = 'lon lat'
    #         xds['QC06'].attrs['grid_mapping'] = 'crs'
    #         rename_qc['QC06'] = 'syntax_qc'

    #     # QC07
    #     if 'QC07' in xds:
    #         xds['QC07'].attrs['long_name'] = 'Maximum Velocity Threshold (QARTOD Test 07) Flag Masks'
    #         xds['QC07'].attrs['valid_range'] = [1, 9]
    #         xds['QC07'].attrs['flag_values'] = [1, 2, 3, 4, 5]
    #         xds['QC07'].attrs['flag_meanings'] = 'pass not_evaluated suspect fail missing_data'
    #         xds['QC07'].attrs['coordinates'] = 'lon lat'
    #         xds['QC07'].attrs['grid_mapping'] = 'crs'
    #         rename_qc['QC07'] = 'max_threshold_qc'

    #     # QC08
    #     if 'QC08' in xds:
    #         xds['QC08'].attrs['long_name'] = 'Valid Location (QARTOD Test 08) Flag Masks'
    #         xds['QC08'].attrs['valid_range'] = [1, 9]
    #         xds['QC08'].attrs['flag_values'] = [1, 2, 3, 4, 5]
    #         xds['QC08'].attrs['flag_meanings'] = 'pass not_evaluated suspect fail missing_data'
    #         xds['QC08'].attrs['coordinates'] = 'lon lat'
    #         xds['QC08'].attrs['grid_mapping'] = 'crs'
    #         rename_qc['QC08'] = 'valid_location_qc'

    #     # QC09
    #     if 'QC09' in xds:
    #         xds['QC09'].attrs['long_name'] = 'Radial Count (QARTOD Test 09) Flag Masks'
    #         xds['QC09'].attrs['valid_range'] = [1, 9]
    #         xds['QC09'].attrs['flag_values'] = [1, 2, 3, 4, 5]
    #         xds['QC09'].attrs['flag_meanings'] = 'pass not_evaluated suspect fail missing_data'
    #         xds['QC09'].attrs['coordinates'] = 'lon lat'
    #         xds['QC09'].attrs['grid_mapping'] = 'crs'
    #         rename_qc['QC09'] = 'radial_count_qc'

    #     # QC10
    #     if 'QC10' in xds:
    #         xds['QC10'].attrs['long_name'] = 'Spatial Median Filter (QARTOD Test 10) Flag Masks'
    #         xds['QC10'].attrs['valid_range'] = [1, 9]
    #         xds['QC10'].attrs['flag_values'] = [1, 2, 3, 4, 5]
    #         xds['QC10'].attrs['flag_meanings'] = 'pass not_evaluated suspect fail missing_data'
    #         xds['QC10'].attrs['coordinates'] = 'lon lat'
    #         xds['QC10'].attrs['grid_mapping'] = 'crs'
    #         rename_qc['QC10'] = 'spatial_median_filter_qc'

    #     # QC11
    #     if 'QC11' in xds:
    #         xds['QC11'].attrs['long_name'] = 'Temporal Gradient (QARTOD Test 11) Flag Masks'
    #         xds['QC11'].attrs['valid_range'] = [1, 9]
    #         xds['QC11'].attrs['flag_values'] = [1, 2, 3, 4, 5]
    #         xds['QC11'].attrs['flag_meanings'] = 'pass not_evaluated suspect fail missing_data'
    #         xds['QC11'].attrs['coordinates'] = 'lon lat'
    #         xds['QC11'].attrs['grid_mapping'] = 'crs'
    #         rename_qc['QC11'] = 'temporal_gradient_qc'

    #     # QC12
    #     if 'QC12' in xds:
    #         xds['QC12'].attrs['long_name'] = 'Average Radial Bearing (QARTOD Test 12) Flag Masks'
    #         xds['QC12'].attrs['valid_range'] = [1, 9]
    #         xds['QC12'].attrs['flag_values'] = [1, 2, 3, 4, 5]
    #         xds['QC12'].attrs['flag_meanings'] = 'pass not_evaluated suspect fail missing_data'
    #         xds['QC12'].attrs['coordinates'] = 'lon lat'
    #         xds['QC12'].attrs['grid_mapping'] = 'crs'
    #         rename_qc['QC12'] = 'average_radial_bearing_qc'

    #     # QC12
    #     if 'PRIM' in xds:
    #         xds['PRIM'].attrs['long_name'] = 'Primary Flag Masks'
    #         xds['PRIM'].attrs['valid_range'] = [1, 9]
    #         xds['PRIM'].attrs['flag_values'] = [1, 2, 3, 4, 5]
    #         xds['PRIM'].attrs['flag_meanings'] = 'pass not_evaluated suspect fail missing_data'
    #         xds['PRIM'].attrs['coordinates'] = 'lon lat'
    #         xds['PRIM'].attrs['grid_mapping'] = 'crs'
    #         rename_qc['PRIM'] = 'primary_flag_qc'

    #     if 'QCOP' in xds:
    #         xds['QCOP'].attrs['long_name'] = 'Operator Flag Masks'
    #         xds['QCOP'].attrs['valid_range'] = [1, 9]
    #         xds['QCOP'].attrs['flag_values'] = [1, 2, 3, 4, 5]
    #         xds['QCOP'].attrs['flag_meanings'] = 'pass not_evaluated suspect fail missing_data'
    #         xds['QCOP'].attrs['coordinates'] = 'lon lat'
    #         xds['QCOP'].attrs['grid_mapping'] = 'crs'
    #         xds['QCOP'].attrs['comment'] = 'Flag that is manually set by operator. Used for flagging vectors that are not detected by QC tests but are clearly wrong.'
    #         rename_qc['QCOP'] = 'operator_flag_qc'


    #     # rename variables to something meaningful if they exist in the xarray dataset
    #     xds = xds.rename({ k: v for k, v in rename_qc.items() if k in xds })
    #     # del xds.attrs['TimeStamp']

    #     return xds

    def file_type(self):
        """Return a string representing the type of file this is."""
        return 'radial'

    # def clean_header(self, split_origin=False):
    #     """
    #     Clean up the radial header dictionary so that you can upload it to the HFR MySQL Database.
    #     :return:
    #     """
    #     keep = ['CTF', 'FileType', 'LLUVSpec', 'UUID', 'Manufacturer', 'Site', 'TimeStamp', 'TimeZone', 'TimeCoverage',
    #             'Origin', 'GreatCircle', 'GeodVersion', 'LLUVTrustData', 'RangeStart', 'RangeEnd', 'RangeResolutionKMeters',
    #             'AntennaBearing', 'ReferenceBearing', 'AngularResolution', 'SpatialResolution', 'PatternType', 'PatternDate',
    #             'PatternResolution', 'TransmitCenterFreqMHz', 'DopplerResolutionHzPerBin', 'FirstOrderMethod',
    #             'BraggSmoothingPoints', 'CurrentVelocityLimits', 'BraggHasSecondOrder', 'RadialBraggPeakDropOff',
    #             'RadialBraggPeakNull', 'RadialBraggNoiseThreshold', 'PatternAmplitudeCorrections', 'PatternPhaseCorrections',
    #             'PatternAmplitudeCalculations', 'PatternPhaseCalculations', 'RadialMusicParameters', 'MergedCount',
    #             'RadialMinimumMergePoints', 'FirstOrderCalc', 'MergeMethod', 'PatternMethod', 'TransmitSweepRateHz',
    #             'TransmitBandwidthKHz', 'SpectraRangeCells', 'SpectraDopplerCells', 'TableType', 'TableColumns', 'TableColumnTypes',
    #             'TableRows', 'TableStart', 'CurrentVelocityLimit']

    #     # TableColumnTypes
    #     key_list = list(self.metadata.keys())
    #     for key in key_list:
    #         if key not in keep:
    #             del self.metadata[key]

    #     for k, v in self.metadata.items():
    #         if 'Site' in k:
    #             # WERA has lines like: '%Site: csw "CSW' and '%Site: gtn "gtn'
    #             # This should work for both CODAR and WERA files
    #             split_site = v.split(' ', 1)[0]
    #             self.metadata[k] = ''.join(e for e in split_site if e.isalnum())
    #         elif k in ('TimeStamp', 'PatternDate'):
    #             try:
    #                 t_list = [int(s) for s in v.split()]
    #                 self.metadata[k] = dt.datetime(*t_list)
    #             except ValueError:
    #                 # Can't parse a date, set to None
    #                 self.metadata[k] = None
    #         elif 'TimeZone' in k:
    #             self.metadata[k] = v.split('"')[1]
    #         elif 'TableColumnTypes' in k:
    #             self.metadata[k] = ' '.join([x.strip() for x in v.strip().split(' ')])
    #         elif 'Origin' in k:
    #             if split_origin:
    #                 self.metadata[k] = re.findall(r"[-+]?\d*\.\d+|\d+", v)
    #             else:
    #                 self.metadata[k] = v.lstrip()
    #         elif k in ('RangeStart', 'RangeEnd', 'AntennaBearing', 'ReferenceBearing', 'AngularResolution', 'SpatialResolution',
    #                    'FirstOrderMethod', 'BraggSmoothingPoints', 'BraggHasSecondOrder', 'MergedCount',
    #                    'RadialMinimumMergePoints', 'FirstOrderCalc', 'SpectraRangeCells', 'SpectraDopplerCells',
    #                    'TableColumns', 'TableRows',  'PatternResolution', 'CurrentVelocityLimit', 'TimeCoverage'):
    #             try:
    #                 self.metadata[k] = int(v)
    #             except ValueError:
    #                 temp = v.split(' ')[0]
    #                 try:
    #                     self.metadata[k] = int(temp)
    #                 except ValueError:
    #                     try:
    #                         self.metadata[k] = int(temp.split('.')[0])
    #                     except ValueError:
    #                         self.metadata[k] = None
    #         elif k in ('RangeResolutionKMeters', 'CTF', 'TransmitCenterFreqMHz', 'DopplerResolutionHzPerBin',
    #                    'RadialBraggPeakDropOff', 'RadialBraggPeakNull', 'RadialBraggNoiseThreshold', 'TransmitSweepRateHz',
    #                    'TransmitBandwidthKHz'):
    #             try:
    #                 self.metadata[k] = float(v)
    #             except ValueError:
    #                 try:
    #                     self.metadata[k] = float(v.split(' ')[0])
    #                 except ValueError:
    #                     self.metadata[k] = None
    #         else:
    #             continue

    #     required = ['Origin', 'TransmitCenterFreqMHz']
    #     present_keys = self.metadata.keys()
    #     for key in required:
    #         if key not in present_keys:
    #             self.metadata[key] = None

    # def create_ruv(self, filename):
    #     """
    #     Create a CODAR Radial (.ruv) file from radial instance
    #     :param filename: User defined filename of radial file you want to save
    #     :return:
    #     """
    #     create_dir(os.path.dirname(filename))
    #     rcopy = copy.deepcopy(self)
    #     with open(filename, 'w') as f:
    #         # Write header
    #         for metadata_key, metadata_value in self.metadata.items():
    #             if 'ProcessedTimeStamp' in metadata_key:
    #                 break
    #             else:
    #                 f.write('%{}: {}\n'.format(metadata_key, metadata_value))

    #         # Write data tables. Anything beyond the first table is commented out.
    #         for table in self._tables.keys():
    #             for table_key, table_value in self._tables[table].items():
    #                 if table_key != 'data':
    #                     if (table_key == 'TableType') & (table == '1'):
    #                         if 'QCTest' in self.metadata:
    #                             f.write('%QCFileVersion: 1.0.0\n')
    #                             f.write('%QCReference: Quality control reference: IOOS QARTOD HF Radar ver 1.0 May 2016\n')
    #                             f.write('%QCFlagDefinitions: 1=pass 2=not_evaluated 3=suspect 4=fail 9=missing_data\n')
    #                             f.write('%QCTestFormat: "test_name [qc_thresholds]: test_result"\n')

    #                             for test in self.metadata['QCTest']:
    #                                 f.write('%QCTest: {}\n'.format(test))
    #                         f.write('%{}: {}\n'.format(table_key, table_value))
    #                     elif table_key == 'TableColumns':
    #                         f.write('%TableColumns: {}\n'.format(len(self._tables[table]['data'].columns)))
    #                     elif table_key == 'TableColumnTypes':
    #                         f.write('%TableColumnTypes: {}\n'.format(' '.join(self._tables[table]['data'].columns.to_list())))
    #                     elif table_key == 'TableStart':
    #                         f.write('%{}: {}\n'.format(table_key, table_value))
    #                     elif table_key == '_TableHeader':
    #                         pass
    #                     else:
    #                         f.write('%{}: {}\n'.format(table_key, table_value))

    #             if 'datetime' in self._tables[table]['data'].keys():
    #                 self._tables[table]['data'] = self._tables[table]['data'].drop(['datetime'], axis=1)

    #             if table == '1':
    #                 # Fill NaN with 999.000 which is the standard fill value for codar lluv filesself._tables[table]['TableColumnTypes']
    #                 self.data = self.data.fillna(999.000)

    #                 try:
    #                     self.data['LOND'] = self.data['LOND'].apply(lambda x: "{:.7f}".format(x))
    #                     self.data['LATD'] = self.data['LATD'].apply(lambda x: "{:.7f}".format(x))
    #                     self.data['ESPC'] = self.data['ESPC'].apply(lambda x: "{:.3f}".format(x))
    #                     if 'ETMP' in self.data.columns:
    #                         self.data['ETMP'] = self.data['ETMP'].apply(lambda x: "{:.3f}".format(x))
    #                     self.data['BEAR'] = self.data['BEAR'].apply(lambda x: "{:.1f}".format(x))
    #                     self.data['HEAD'] = self.data['HEAD'].apply(lambda x: "{:.1f}".format(x))
    #                 except:
    #                     self = rcopy
    #                     print("Unexpected error in formatting one of these columns: LOND LATD ESPC ETMP BEAR HEAD")

    #                 # Convert _TableHeader to a new dataframe and concatenate to dataframe containing radial data
    #                 # This allows for the output format to follow CODARS CTF specifications
    #                 row_df = pd.DataFrame([self._tables['1']['_TableHeader'][1]], columns=self._tables['1']['_TableHeader'][0])
    #                 self.data.columns = self._tables['1']['_TableHeader'][0]
    #                 self.data = pd.concat([row_df, self.data], ignore_index=True)
    #                 self.data.insert(0, '%%', np.nan)  # Insert column at the beginning of dataframe of NaNs
    #                 self.data.iloc[0, self.data.columns.get_loc('%%')] = '%%'  # make the first row in the first column a '%%'

    #                 # Output data table to string
    #                 #self.data.to_string(f, index=False, justify='center', header=True, na_rep=' ')
    #                 self.data.temp = re.sub(' %%', '%%', self.data.to_string(index=False, justify='right', header=True, na_rep=' '))
    #                 f.write(self.data.temp)
    #             else:
    #                 self._tables[table]['data'].insert(0, '%%', '%')
    #                 self._tables[table]['data'] = self._tables[table]['data'].fillna(999.000)
    #                 self._tables[table]['data'].to_string(f, index=False, justify='center', header=True)

    #             if int(table) > 1:
    #                 f.write('\n%TableEnd: {}\n'.format(table))
    #             else:
    #                 f.write('\n%TableEnd: \n')
    #             f.write('%%\n')

    #         # Write footer containing processing information
    #         f.write('%ProcessedTimeStamp: {}\n'.format(self.metadata['ProcessedTimeStamp']))
    #         for tool in self.metadata['ProcessingTool']:
    #             f.write('%ProcessingTool: {}\n'.format(tool))
    #             # f.write('%{}: {}\n'.format(footer_key, footer_value))
    #         f.write('%End:')

    # def export(self, filename, user_attributes, file_type='radial', ):
    #     """
    #     Export radial file as either a codar .ruv file or a netcdf .nc file
    #     :param filename: User defined filename of radial file you want to save
    #     :param file_type: Type of file to export radial: radial (default) or netcdf
    #     :return:
    #     """

    #     if not self.is_valid():
    #         raise ValueError("Could not export ASCII data, the input file was invalid.")

    #     if os.path.isfile(filename):
    #         os.remove(filename)

    #     if file_type == 'radial':
    #         self.create_ruv(filename)
    #     elif 'netcdf' in file_type:
    #         if '.nc' in filename:
    #             save_file = filename
    #         else:
    #             save_file = filename + '.nc'
    #         self.create_netcdf(save_file, user_attributes, nc_shape=file_type, enhance=True)

    def initialize_qc(self):
        """
        Initialize dictionary entry for QC metadata.
        """
        # Initialize dictionary entry for QC metadta
        self.metadata['QCTest'] = {}
        

    # EUROPEAN HFR NODE (EHN) QC Tests
    
    def qc_ehn_avg_radial_bearing(self, minBear=0, maxBear=360):
        """
        
        This QC test labels radial velocity vectors with a ‘good_data” flag if the average radial bearing 
        of all the vectors contained in the radial lie within the specified range for normal operations.
        Otherwise, the vectors are labeled with a “bad_data” flag.
        The ARGO QC flagging scale is used.
        
        This test is applicable only to DF systems. 
        Data files from BF systems will have this variable filled with a “good_data” flag.
        
        This test was defined in the framework of the EuroGOOS HFR Task Team based on the
        Average Radial Bearing test (QC207) from the Integrated Ocean Observing System (IOOS) 
        Quality Assurance of Real-Time Oceanographic Data (QARTOD).
        
        INPUTS:
            minBear: minimum angle in degrees of the specified range for normal operations
            maxBear: maximum angle in degrees of the specified range for normal operations
            
        
        """
        # Set the test name
        testName = 'AVRB_QC'
        
        # Evaluate the average bearing
        if self.is_wera:
            self.data.loc[:,testName] = 1
        else:
            avgBear = self.data['BEAR'].mean()
    
            if avgBear >= minBear and avgBear <= maxBear:
                self.data.loc[:,testName] = 1
            else:
                self.data.loc[:,testName] = 4

        self.metadata['QCTest'][testName] = 'Average Radial Bearing QC Test - Test applies to entire file. ' \
            + 'Thresholds=[' + f'minimum bearing={minBear} (degrees) - ' + f'maximum bearing={maxBear} (degrees)]'
        
    def qc_ehn_radial_count(self, radMinCount=150):
        """
        
        This test labels radial velocity vectors with a “good data” flag if the number of velocity vectors contained
        in the radial is bigger than the minimum count specified for normal operations.
        Otherwise the vectors are labeled with a “bad data” flag.
        The ARGO QC flagging scale is used.
        
        This test was defined in the framework of the EuroGOOS HFR Task Team based on the
        Radial Count test from (QC204) the Integrated Ocean Observing System (IOOS) Quality Assurance of 
        Real-Time Oceanographic Data (QARTOD).
        
        INPUTS:
            radMinCount: minimum number of velocity vectors for normal operations            
        
        """
        # Set the test name
        testName = 'RDCT_QC'
        
        # Get the number of velocity vectors contained in the radial
        numRad = len(self.data)

        if numRad >= radMinCount:
            self.data.loc[:,testName] = 1
        else:
            self.data.loc[:,testName] = 4

        self.metadata['QCTest'][testName] = 'Radial Count QC Test - Test applies to entire file. ' \
            + 'Threshold=[' + f'minimum number of radial vectors={radMinCount}]'
        
    def qc_ehn_maximum_velocity(self, radMaxSpeed=1.2):
        """
        This test labels radial velocity vectors whose module is smaller than a maximum velocity threshold 
        with a “good data” flag. Otherwise the vectors are labeled with a “bad data” flag.
        The ARGO QC flagging scale is used.
        
        This test was defined in the framework of the EuroGOOS HFR Task Team based on the
        Max Threshold test (QC202) from the Integrated Ocean Observing System (IOOS) Quality Assurance of 
        Real-Time Oceanographic Data (QARTOD).
        
        INPUTS:
            radMaxSpeed: maximum velocity in m/s for normal operations                     
        """
        # Set the test name
        testName = 'CSPD_QC'
    
        # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
        self.data.loc[:,testName] = 1
    
        # set bad flag for velocities not passing the test
        if self.is_wera:
            self.data.loc[(self.data['VELO'].abs() > radMaxSpeed), testName] = 4          # velocity in m/s (CRAD)
        else:
            self.data.loc[(self.data['VELO'].abs() > radMaxSpeed*100), testName] = 4      # velocity in cm/s (LLUV)
    
        self.metadata['QCTest'][testName] = 'Velocity Threshold QC Test - Test applies to each vector. ' \
            + 'Threshold=[' + f'maximum velocity={radMaxSpeed} (m/s)]'
        
    def qc_ehn_median_filter(self, dLim=10, curLim=0.5):
        """
        This test evaluates, for each radial vector, the median of the modules of all vectors lying
        within a distance of <dLim> km. 
        Each vector for which the difference between its module and the median is smaller than
        the specified threshold for normal operations (curLim), is labeled with a "good data" flag.
        Otherwise the vector is labeled with a “bad data” flag.
        The ARGO QC flagging scale is used.
        The native CRS of the Radial is used for distance calculations.
        
        This test was defined in the framework of the EuroGOOS HFR Task Team based on the 
        Spatial Median test (QC205) from the Integrated Ocean Observing System (IOOS) Quality 
        Assurance of Real-Time Oceanographic Data (QARTOD).
        
        INPUTS:
            dLim: distance limit in km for selecting vectors for median calculation
            curLim: velocity-median difference threshold in m/s for normal opertions
        """
        # Set the test name
        testName = 'MDFL_QC'
        
        # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
        self.data.loc[:,testName] = 1
        
        # Get the ellipsoid of the radial coordinate reference system
        radEllps = self.metadata['GreatCircle'].split()[0].replace('"','')
        
        # Create Geod object with the retrieved ellipsoid
        g = Geod(ellps=radEllps)
        
        # Evaluate the median of velocities within dLim for each radial vector
        median = self.data.loc[:,['LOND','LATD']].apply(lambda x: velocityMedianInDistLimits(x,self.data,dLim,g),axis=1)
        
        # set bad flag for vectors not passing the test
        if self.is_wera:
            self.data.loc[abs(self.data['VELO'] - median) > curLim, testName] = 4          # velocity in m/s (CRAD)
        else:
            self.data.loc[abs(self.data['VELO'] - median) > curLim*100, testName] = 4      # velocity in cm/s (LLUV)
        
        self.metadata['QCTest'][testName] = 'Median Filter QC Test - Test applies to each vector. ' \
            + 'Thresholds=[' + f'distance limit={str(dLim)} (km) ' + f'velocity-median difference threshold={str(curLim)} (m/s)]'
        
    def qc_ehn_over_water(self):
        """
        This test labels radial velocity vectors that lie on water with a good data” flag.
        Otherwise the vectors are labeled with a “bad data” flag.
        The ARGO QC flagging scale is used.
        
        Radial vector coordinates are checked against a reference file containing information 
        about which locations are over land or in an unmeasurable area (for example, behind an 
        island or point of land). 
        The GeoPandas "naturalearth_lowres" is used as reference.
        CODAR radials are beforehand flagged based on the "VFLAG" variable (+128 means "on land").
        
        This test was defined in the framework of the EuroGOOS HFR Task Team based on the 
        Valid Location (Test 203) from the Integrated Ocean Observing System (IOOS) Quality 
        Assurance of Real-Time Oceanographic Data (QARTOD).
        """
        # Set the test name
        testName = 'OWTR_QC'
        
        # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
        self.data.loc[:,testName] = 1
        
        # Set the "vector-on-land" flag name for CODAR data
        vectorOnLandFlag = 'VFLG'

        # Set bad flag where land is flagged by CODAR manufacturer
        if not self.is_wera:
            if vectorOnLandFlag in self.data:
                self.data.loc[(self.data[vectorOnLandFlag] == 128), testName] = 4  
            
        # Set bad flag where land is flagged (mask_over_land method)
        self.data.loc[~self.mask_over_land(subset=False), testName] = 4
            
        self.metadata['QCTest'][testName] = 'Over Water QC Test - Test applies to each vector. ' \
            + 'Thresholds=[' + 'GeoPandas "naturalearth_lowres"]'
        
    def qc_ehn_maximum_variance(self, radMaxVar=1):
        """
        This test labels radial velocity vectors whose temporal variance is smaller than
        a maximum variance threshold with a “good data” flag. 
        Otherwise the vectors are labeled with a “bad data” flag.
        The ARGO QC flagging scale is used.
        
        This test was defined in the framework of the EuroGOOS HFR Task Team based on the
        U Component Uncertainty and V Component Uncertainty tests (QC306 and QC307) from the
        Integrated Ocean Observing System (IOOS) Quality Assurance of Real-Time Oceanographic 
        Data (QARTOD).
        
        This test is NOT RECOMMENDED for CODAR data because the parameter defining the variance
        is computed at each time step, and therefore considered not statistically solid 
        (as documented in the fall 2013 CODAR Currents Newsletter).
        
        INPUTS:
            radMaxVar: maximum variance in m2/s2 for normal operations                     
        """
        # Set the test name
        testName = 'VART_QC'
    
        # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
        self.data.loc[:,testName] = 1
    
        # Set bad flag for variances not passing the test
        if self.is_wera:
            self.data.loc[(self.data['HCSS'] > radMaxVar), testName] = 4           # HCSS is the temporal variance for WERA data
        else:
            self.data.loc[((self.data['ETMP']/100)**2 > radMaxVar), testName] = 4  # ETMP is the temporal standard deviation in cm/s for CODAR data
    
        self.metadata['QCTest'][testName] = 'Variance Threshold QC Test - Test applies to each vector. ' \
            + 'Threshold=[' + f'maximum variance={radMaxVar} (m2/s2)]'
        
    def qc_ehn_temporal_derivative(self, r0, tempDerThr=1):
        """
        This test compares the velocity of each radial vector with the velocity of the radial vector 
        measured in the previous timestamp at the same location.
        Each vector for which the velocity difference is smaller than the specified threshold for normal 
        operations (tempDerThr), is labeled with a "good data" flag.
        Otherwise the vector is labeled with a “bad data” flag.
        The ARGO QC flagging scale is used.
        
        This test was defined in the framework of the EuroGOOS HFR Task Team based on the 
        Temporal Gradient test (QC206) from the Integrated Ocean Observing System (IOOS) Quality 
        Assurance of Real-Time Oceanographic Data (QARTOD).
        
        INPUTS:
            r0: Radial object of the previous timestamp
            tempDerThr: velocity difference threshold in m/s for normal operations
        """
        # Set the test name
        testName = 'VART_QC'
        
        # Check if the previous timestamp radial file exists
        if not r0 is None:
            # Merge the data DataFrame of the two Radials and evaluate velocity differences at each location
            mergedDF = self.data.merge(r0.data, on=['LOND', 'LATD'], how='left', suffixes=(None, '_x'), indicator='Exist')
            velDiff = (mergedDF['VELO'] - mergedDF['VELO_x']).abs()

            # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
            self.data.loc[:,testName] = 1

            # Set rows of the DataFrame for QC data as not evaluated (flag = 0) for locations existing in the current radial but not in the previous one
            self.data.loc[mergedDF['Exist'] == 'left_only', testName] = 0

            # Set bad flag for vectors not passing the test
            if self.is_wera:
                self.data.loc[(velDiff > tempDerThr), testName] = 4             # velocity in m/s (CRAD)
            else:
                self.data.loc[(velDiff > tempDerThr*100), testName] = 4         # velocity in cm/s (LLUV)

        else:
            # Add new column to the DataFrame for QC data by setting every row as not evaluated (flag = 0)
            self.data.loc[:,testName] = 0
        
        self.metadata['QCTest'][testName] = 'Temporal Derivative QC Test - Test applies to each vector. ' \
            + 'Threshold=[' + f'velocity difference threshold={str(tempDerThr)} (m/s)]'
        
    def qc_ehn_overall_qc_flag(self):
        """
        
        This QC test labels radial velocity vectors with a ‘good_data” flag if all QC tests are passed.
        Otherwise, the vectors are labeled with a “bad_data” flag.
        The ARGO QC flagging scale is used.
        
        INPUTS:
            
        
        """
        # Set the test name
        testName = 'QCflag'
        
        # Add new column to the DataFrame for QC data by setting every row as not passing the test (flag = 4)
        self.data.loc[:,testName] = 4
        
        # Set good flags for vectors passing all QC tests
        self.data.loc[self.data.loc[:, self.data.columns.str.contains('_QC')].eq(1).all(axis=1), testName] = 1

        self.metadata['QCTest'][testName] = 'Overall QC Flag - Test applies to each vector. Test checks if all QC tests are passed.'
            
        
    # QARTOD QC TESTS
        
    # def qc_qartod_avg_radial_bearing(self, reference_bearing, warning_threshold=15, failure_threshold=30):
    #     """
    #     Integrated Ocean Observing System (IOOS) Quality Assurance of Real-Time Oceanographic Data (QARTOD)
    #     Valid Location (Test 207)
    #     Check that the average radial bearing remains relatively constant (Roarty et al. 2012).

    #     It is expected that the average of all radial velocity bearings AVG_RAD_BEAR obtained during a sample
    #     interval (e.g., 1 hour) should be close to a reference bearing REF_RAD_BEAR and not vary beyond warning
    #     or failure thresholds.
    #     :return:
    #     """
    #     test_str = 'QC207'
    #     # Absolute value of the difference between the bearing mean and reference bearing
    #     absolute_difference = np.abs(self.data['BEAR'].mean() - reference_bearing)

    #     if absolute_difference >= failure_threshold:
    #         flag = 4
    #     elif (absolute_difference >= warning_threshold) & (absolute_difference < failure_threshold):
    #         flag = 3
    #     elif absolute_difference < warning_threshold:
    #         flag = 1

    #     self.data[test_str] = flag  # Assign the flags to the column
    #     self.metadata['QCTest'].append((
    #         f'qc_qartod_avg_radial_bearing ({test_str}) - Test applies to entire file. Thresholds='
    #         '[ '
    #         f'reference_bearing={reference_bearing} (degrees) '
    #         f'warning={warning_threshold} (degrees) '
    #         f'failure={failure_threshold} (degrees) '
    #         f']: See result in column {test_str} below'
    #     ))
    #     self.append_to_tableheader(test_str, '(flag)')

    # def qc_qartod_valid_location(self):
    #     """
    #     Integrated Ocean Observing System (IOOS) Quality Assurance of Real-Time Oceanographic Data (QARTOD)
    #     Valid Location (Test 203)
    #     Removes radial vectors placed over land or in other unmeasureable areas

    #     Radial vector coordinates are checked against a reference file containing information about which locations
    #     are over land or in an unmeasurable area (for example, behind an island or point of land). Radials in these
    #     areas will be flagged with a code (FLOC) in the radial file (+128 in CODAR radial files) and are not included
    #     in total vector calculations.

    #     Link: https://ioos.noaa.gov/ioos-in-action/manual-real-time-quality-control-high-frequency-radar-surface-current-data/
    #     :return:
    #     """
    #     test_str = 'QC203'
    #     flag_column = 'VFLG'

    #     if flag_column in self.data:
    #         self.data[test_str] = 1  # add new column of passing values
    #         self.data.loc[(self.data[flag_column] == 128), test_str] = 4  # set value equal to 4 where land is flagged (manufacturer)
    #         self.data.loc[~self.mask_over_land(subset=False), test_str] = 4  # set value equal to 4 where land is flagged (mask_over_land)
    #         self.metadata['QCTest'].append((
    #             f'qc_qartod_valid_location ({test_str}) - Test applies to each row. Thresholds=[{flag_column}==128]: '
    #             f'See results in column {test_str} below'
    #         ))
    #         self.append_to_tableheader(test_str, '(flag)')

    #     else:
    #         logger.warning(f"qc_qartod_valid_location not run, no {flag_column} column")

    # def qc_qartod_radial_count(self, radial_min_count=150, radial_low_count=300):
    #     """
    #     Integrated Ocean Observing System (IOOS) Quality Assurance of Real-Time Oceanographic Data (QARTOD)
    #     Radial Count (Test 204)
    #     Rejects radials in files with low radial counts (poor radial map coverage).

    #     The number of radials (RCNT) in a radial file must be above a threshold value RCNT_MIN to pass the test and
    #     above a value RC_LOW to not be considered suspect. If the number of radials is below the minimum level,
    #     it indicates a problem with data collection. In this case, the file should be rejected and none of the
    #     radials used for total vector processing.

    #     Link: https://ioos.noaa.gov/ioos-in-action/manual-real-time-quality-control-high-frequency-radar-surface-current-data/
    #     :param min_radials: Minimum radial count threshold below which the file should be rejected. min_radials < low_radials
    #     :param low_radials: Low radial count threshold below which the file should be considered suspect. low_radials > min_radials
    #     :return:
    #     """
    #     test_str = 'QC204'
    #     column_flag = 'VFLG'

    #     # If a vector flag is supplied by the vendor, subset by that first
    #     if column_flag in self.data:
    #         num_radials = len(self.data[self.data[column_flag] != 128])
    #     else:
    #         num_radials = len(self.data)

    #     if num_radials < radial_min_count:
    #         radial_count_flag = 4
    #     elif (num_radials >= radial_min_count) and (num_radials <= radial_low_count):
    #         radial_count_flag = 3
    #     elif num_radials > radial_low_count:
    #         radial_count_flag = 1

    #     self.data[test_str] = radial_count_flag
    #     self.metadata['QCTest'].append((
    #         f'qc_qartod_radial_count ({test_str}) - Test applies to entire file. Thresholds='
    #         '[ '
    #         f'failure={radial_min_count} (radials) '
    #         f'warning_num={radial_low_count} (radials) '
    #         f'<valid_radials={num_radials}> '
    #         f']:  See results in column {test_str} below'
    #     ))
    #     self.append_to_tableheader(test_str, '(flag)')

    # def qc_qartod_maximum_velocity(self, radial_max_speed=250, radial_high_speed=150):
    #     """
    #     Integrated Ocean Observing System (IOOS) Quality Assurance of Real-Time Oceanographic Data (QARTOD)
    #     Max Threshold (Test 202)
    #     Ensures that a radial current speed is not unrealistically high.

    #     The maximum radial speed threshold (RSPDMAX) represents the maximum reasonable surface radial velocity
    #     for the given domain.

    #     Link: https://ioos.noaa.gov/ioos-in-action/manual-real-time-quality-control-high-frequency-radar-surface-current-data/
    #     :param threshold: Maximum Radial Speed (cm/s)
    #     :return:
    #     """
    #     test_str = 'QC202'

    #     self.data['VELO'] = self.data['VELO'].astype(float)  # make sure VELO is a float

    #     # Add new column to dataframe for test, and set every row as passing, 1, flag
    #     self.data[test_str] = 1

    #     # velocity is less than radial_max_speed but greater than radial_high_speed, set that row as a warning, 3, flag
    #     self.data.loc[(self.data['VELO'].abs() < radial_max_speed) & (self.data['VELO'].abs() > radial_high_speed), test_str] = 3

    #     # if velocity is greater than radial_max_speed, set that row as a fail, 4, flag
    #     self.data.loc[(self.data['VELO'].abs() > radial_max_speed), test_str] = 4

    #     self.metadata['QCTest'].append((
    #         f'qc_qartod_maximum_velocity ({test_str}) - Test applies to each row. Thresholds='
    #         '[ '
    #         f'high_vel={str(radial_high_speed)} (cm/s) '
    #         f'max_vel={str(radial_max_speed)} (cm/s) '
    #         f']: See results in column {test_str} below'
    #     ))
    #     self.append_to_tableheader(test_str, '(flag)')

    # def qc_qartod_spatial_median(self, radial_smed_range_cell_limit=2.1, radial_smed_angular_limit=10, radial_smed_current_difference=30):
    #     """
    #     Integrated Ocean Observing System (IOOS) Quality Assurance of Real-Time Oceanographic Data (QARTOD)
    #     Spatial Median (Test 205)
    #     Ensures that the radial velocity is not too different from nearby radial velocities.
    #     RV is the radial velocity
    #     NV is a set of radial velocities for neighboring radial cells (cells within radius of 'radial_smed_range_cell_limit' * Range Step (km)
    #     and whose vector bearing (angle of arrival at site) is also within 'radial_smed_angular_limit' degrees of the source vector's bearing)
    #     Required to pass the test: |RV - median(NV)| <= radial_smed_current_difference
    #     Link: https://ioos.noaa.gov/ioos-in-action/manual-real-time-quality-control-high-frequency-radar-surface-current-data/
    #     :param RCLim: multiple of range step which depends on the radar type
    #     :param AngLim: limit for number of degrees from source radial's bearing (degrees)
    #     :param CurLim: Current difference radial_smed_current_difference (cm/s)
    #     :return:
    #     """
    #     test_str = 'QC205'

    #     self.data[test_str] = 1
    #     try:
    #         Rstep = float(self.metadata['RangeResolutionKMeters'])
    #         # Rstep = np.floor(min(np.diff(np.unique(self.data['RNGE'])))) #use as backup method if other fails?

    #         Bstep = [float(s) for s in re.findall(r'-?\d+\.?\d*', self.metadata['AngularResolution'])]
    #         Bstep = Bstep[0]
    #         # Bstep = int(min(np.diff(np.unique(self.data['BEAR']))))  #use as backup method if other fails?

    #         RLim = int(round(radial_smed_range_cell_limit))  # if not an integer will cause an error later on
    #         BLim = int(radial_smed_angular_limit / Bstep)  # if not an integer will cause an error later on

    #         # convert bearing into bearing cell numbers
    #         adj = np.mod(min(self.data['BEAR']),Bstep)
    #         Bcell = ((self.data['BEAR'] - adj) / Bstep) - 1
    #         Bcell = Bcell.astype(int)
    #         # Btable = np.column_stack((self.data['BEAR'], Bcell))  #only for debugging

    #         # convert range into range cell numbers
    #         Rcell = (np.floor((self.data['RNGE'] / Rstep) + 0.1))
    #         Rcell = Rcell - min(Rcell)
    #         Rcell = Rcell.astype(int)
    #         # Rtable = np.column_stack((self.data['RNGE'], Rcell))   #only for debugging
    #         Rcell = self.data['SPRC']

    #         # place velocities into a matrix with rows defined as bearing cell# and columns as range cell#
    #         BRvel = np.zeros((int(360 / Bstep), max(Rcell) + 1), dtype=int) + np.nan
    #         BRind = np.zeros((int(360 / Bstep), max(Rcell) + 1), dtype=int) + np.nan

    #         for xx in range(len(self.data['VELO'])):
    #             BRvel[Bcell[xx]][Rcell[xx]] = self.data['VELO'][xx]
    #             BRind[Bcell[xx]][Rcell[xx]] = xx  # keep track of indices so easier to return to original format

    #         # deal with 359 to 0 transition in bearing by
    #         # repeating first BLim rows at the bottom and last BLim rows at the top
    #         # also pad ranges with NaNs by adding extra columns on the left and right of the array
    #         # this keeps the indexing for selecting the neighbors from breaking

    #         BRtemp = np.append(np.append(BRvel[-BLim:], BRvel, axis=0), BRvel[:BLim], axis=0)
    #         rangepad = np.zeros((BRtemp.shape[0], RLim), dtype=int) + np.nan
    #         BRpad = np.append(np.append(rangepad, BRtemp, axis=1), rangepad, axis=1)

    #         # calculate median of neighbors (neighbors include the point itself)
    #         BRmed = BRpad + np.nan  # initialize with an array of NaN
    #         for rr in range(RLim, BRvel.shape[1] + RLim):
    #             for bb in range(BLim, BRvel.shape[0] + BLim):
    #                 temp = BRpad[bb - BLim:bb + BLim + 1, rr - RLim:rr + RLim + 1]  # temp is the matrix of neighbors
    #                 BRmed[bb][rr] = np.nanmedian(temp)

    #         # now remove the padding from the array containing the median values
    #         BRmedtrim = BRmed[BLim:-BLim, RLim:-RLim]

    #         # calculate velocity minus median of neighbors
    #         # and put back into single column using the indices saved in BRind
    #         BRdiff = BRvel - BRmedtrim  # velocity minus median of neighbors, test these values against current radial_smed_current_difference
    #         diffcol = self.data['RNGE'] + np.nan  # initialize a single column for the difference results
    #         for rr in range(BRdiff.shape[1]):
    #             for bb in range(BRdiff.shape[0]):
    #                 if not (np.isnan(BRind[bb][rr])):
    #                     diffcol[BRind[bb][rr]] = BRdiff[bb][rr]
    #         boolean = diffcol.abs() > radial_smed_current_difference

    #         # Another method would take the median of data from any radial cells within a certain
    #         # distance (radius) of the radial being tested.  This method, as coded below, was very slow!
    #         # Perhaps there is a better way to write the code.
    #         # dist contains distance between each location and the other locations
    #         # for rvi in range(len(self.data['VELO'])):
    #         #     dist = np.zeros((len(self.data['LATD']), 1))
    #         #     for i in range(len(self.data['LATD'])):
    #         #         rvloc = self.data['LATD'][i],self.data['LOND'][i]
    #         #         dist[i][0] = distance.distance((self.data['LATD'][rvi],self.data['LOND'][rvi]),(self.data['LATD'][i],self.data['LOND'][i])).kilometers

    #     except TypeError:
    #         diffcol = diffcol.astype(float)
    #         boolean = diffcol.abs() > radial_smed_current_difference

    #     self.data[test_str] = self.data[test_str].where(~boolean, other=4)
    #     self.metadata['QCTest'].append((
    #         f'qc_qartod_spatial_median ({test_str}) - Test applies to each row. Thresholds='
    #         '[ '
    #         f'range_cell_limit={str(radial_smed_range_cell_limit)} (range cells) '
    #         f'angular_limit={str(radial_smed_angular_limit)} (degrees) '
    #         f'current_difference={str(radial_smed_current_difference)} (cm/s) '
    #         f']: See results in column {test_str} below'
    #     ))
    #     self.append_to_tableheader(test_str, '(flag)')

    # def qc_qartod_syntax(self):
    #     """
    #     Integrated Ocean Observing System (IOOS) Quality Assurance of Real-Time Oceanographic Data (QARTOD)
    #     Syntax (Test 201)

    #     This test is required to be QARTOD compliant.

    #     A collection of tests ensuring proper formatting and existence of fields within a radial file.

    #     The radial file may be tested for proper parsing and content, for file format (hfrweralluv1.0, for example),
    #     site code, appropriate time stamp, site coordinates, antenna pattern type (measured or ideal, for DF
    #     systems), and internally consistent row/column specifications.

    #     ----------------------------------------------------------------------------------------------------------------------
    #     Fail: One or more fields are corrupt or contain invalid data, If “File Format” ≠ “hfrweralluv1.0”, flag = 4

    #     Pass: Applies for test pass condition.
    #     ----------------------------------------------------------------------------------------------------------------------
    #     Link: https://ioos.noaa.gov/ioos-in-action/manual-real-time-quality-control-high-frequency-radar-surface-current-data/
    #     :param threshold: Maximum Radial Speed (cm/s)
    #     :return:
    #     """
    #     test_str = 'QC201'

    #     i = 0

    #     # check for timestamp in filename
    #     result = re.search('\d{4}_\d{2}_\d{2}_\d{4}', self.file_name)
    #     if result:
    #         timestr = dt.datetime.strptime(result.group(), '%Y_%m_%d_%H%M')
    #         i = i + 1

    #     # Radial tables must not be empty
    #     if self.is_valid():
    #         i = i + 1

    #     # The following metadata must be defined.
    #     if self.metadata['FileType'] and self.metadata['Site'] and self.metadata['TimeStamp'] and self.metadata['Origin'] and self.metadata['PatternType'] and self.metadata['TimeZone']:
    #         filetime = dt.datetime(*map(int, self.metadata['TimeStamp'].split()))
    #         i = i + 1

    #     # Filename timestamp must match the timestamp reported within the file.
    #     if timestr == filetime:
    #         i = i + 1

    #     # Radial data table columns stated must match the number of columns reported for each row
    #     if len(self._tables['1']['TableColumnTypes'].split()) == self.data.shape[1]:
    #         i = i + 1

    #     # Make sure site location is within range: -180 <= lon <= 180 & -90 <= lat <= 90
    #     latlon = re.findall(r"[-+]?\d*\.\d+|\d+", self.metadata['Origin'])
    #     if (-180 <= float(latlon[1]) <= 180) & (-90 <= float(latlon[0]) <= 90):
    #         i = i + 1

    #     if i == 6:
    #         syntax = 1
    #     else:
    #         syntax = 4

    #     self.data[test_str] = syntax
    #     self.metadata['QCTest'].append(f'qc_qartod_syntax ({test_str}) - Test applies to entire file. Thresholds=[N/A]: See results in column {test_str} below')
    #     self.append_to_tableheader(test_str, '(flag)')

    # def qc_qartod_temporal_gradient(self, r0, gradient_temp_fail=54, gradient_temp_warn=36):
    #     """
    #     Integrated Ocean Observing System (IOOS) Quality Assurance of Real-Time Oceanographic Data (QARTOD)
    #     Temporal Gradient (Test 206)
    #     Checks for satisfactory temporal rate of change of radial components

    #     Test determines whether changes between successive radial velocity measurements at a particular range
    #     and bearing cell are within an acceptable range. GRADIENT_TEMP = |Rt-1 - Rt|

    #     Flags Condition Codable Instructions
    #     Fail = 4 The temporal change between successive radial velocities exceeds the gradient failure threshold.

    #     If GRADIENT_TEMP ≥ GRADIENT_TEMP_FAIL,
    #     flag = 4

    #     Suspect = 3 The temporal change between successive radial velocities is less than the gradient failure threshold but exceeds the gradient warn threshold.
        
    #     If GRADIENT_TEMP < GRADIENT_TEMP_FAIL & GRADIENT_TEMP ≥ GRADIENT_TEMP_WARN,
    #     flag = 3

    #     Pass = 1 The temporal change between successive radial velocities is less than the gradient warn threshold.

    #     If GRADIENT_TEMP < GRADIENT_TEMP_WARN,
    #     flag = 1

    #     Link: https://ioos.noaa.gov/ioos-in-action/manual-real-time-quality-control-high-frequency-radar-surface-current-data/
    #     :param r0: Full path to the filename of the previous hourly radial.
    #     :param gradient_temp_fail: Maximum Radial Speed (cm/s)
    #     :param gradient_temp_warn: Warning Radial Speed (cm/s)
    #     :return:
    #     """
    #     test_str = 'QC206'
    #     # self.data[test_str] = data
    #     self.metadata['QCTest'].append((
    #         f'qc_qartod_temporal_gradient ({test_str}) - Test applies to each row. Thresholds='
    #         '[ '
    #         f'gradient_temp_warn={str(gradient_temp_warn)} (cm/s*hr) '
    #         f'gradient_temp_fail={str(gradient_temp_fail)} (cm/s*hr) '
    #         f']: See results in column {test_str} below'
    #     ))
    #     self.append_to_tableheader(test_str, '(flag)')

    #     if os.path.exists(r0):
    #         r0 = Radial(r0)

    #         merged = self.data.merge(r0.data, on=['LOND', 'LATD'], how='left', suffixes=(None, '_x'), indicator='Exist')
    #         difference = (merged['VELO'] - merged['VELO_x']).abs()

    #         # Add new column to dataframe for test, and set every row as passing, 1, flag
    #         self.data[test_str] = 1

    #         # If any point in the recent radial does not exist in the previous radial, set row as a not evaluated, 2, flag
    #         self.data.loc[merged['Exist'] == 'left_only', test_str] = 2

    #         # velocity is less than radial_max_speed but greater than radial_high_speed, set row as a warning, 3, flag
    #         self.data.loc[(difference < gradient_temp_fail) & (difference > gradient_temp_warn), test_str] = 3

    #         # if velocity is greater than radial_max_speed, set that row as a fail, 4, flag
    #         self.data.loc[(difference > gradient_temp_fail), test_str] = 4

    #     else:
    #         # Add new column to dataframe for test, and set every row as not_evaluated, 2, flag
    #         self.data[test_str] = 2
    #         logging.warning('{} does not exist at specified location. Setting column {} to not_evaluated flag'.format(r0, test_str))

    # def qc_qartod_primary_flag(self, include=None):
    #     """
    #      A primary flag is a single flag set to the worst case of all QC flags within the data record.
    #     :param include: list of quality control tests which should be included in the primary flag
    #     :return:
    #     """
    #     test_str = 'PRIM'

    #     # Set summary flag column all equal to 1
    #     self.data[test_str] = 1

    #     # generate dictionary of executed qc tests found in the header
    #     executed = dict()
    #     for b in [x.split('-')[0].strip() for x in self.metadata['QCTest']]:
    #         i = b.split(' ')
    #         executed[i[0]] = re.sub(r'[()]', '', i[1])

    #     if include:
    #         # only add qartod tests which were set by user to executed dictionary
    #         included_tests = list({key: value for key, value in executed.items() if key in include}.values())
    #     else:
    #         included_tests = list(executed.values())

    #     equals_3 = self.data[included_tests].eq(3).any(axis=1)
    #     self.data[test_str] = self.data[test_str].where(~equals_3, other=3)

    #     equals_4 = self.data[included_tests].eq(4).any(axis=1)
    #     self.data[test_str] = self.data[test_str].where(~equals_4, other=4)

    #     included_test_strs = ', '.join(included_tests)
    #     self.metadata['QCTest'].append((f'qc_qartod_primary_flag ({test_str}) - Primary Flag - Highest flag value of {included_test_strs} ("not_evaluated" flag results ignored)'))
    #     self.append_to_tableheader(test_str, '(flag)')
    #     # %QCFlagDefinitions: 1=pass 2=not_evaluated 3=suspect 4=fail 9=missing_data

    # def append_to_tableheader(self, test_string, test_unit):
    #     self._tables['1']['_TableHeader'][0].append(test_string)
    #     self._tables['1']['_TableHeader'][1].append(test_unit)

    # def reset(self):
    #     logging.info('Resetting instance data variable to original dataset')
    #     self._tables['1']
    #     self.data = self._data_backup
        
