from calendar import EPOCH
import datetime as dt
import pandas as pd
import re
from pyproj import Geod
from shapely.geometry import Point
import xarray as xr
from common import fileParser
# from nc import make_encoding
import numpy as np
import os
from pathlib import Path
from collections import OrderedDict
import geopandas as gpd
import json


import logging

logger = logging.getLogger(__name__)

def addSiteSources(obj,stationData):
    """
    This function adds a DataFrame named site_source to the Waves object containing the codes and the
    lon/lat position of the sites belonging to the network.
    
    INPUTS:
        obj: Waves object
        stationData: DataFrame containing the information of the radial sites that produced the wave data

        
    OUTPUTS:
        obj = Wave object with DataFrame containing the information of the radial sites that produced the wave data
        
    """
    obj.site_source = stationData[['station_id', 'network_id', 'site_lon', 'site_lat', 'transmit_central_frequency']].copy()
    
    return obj


def concat(wave_list, enhance=False):
    """
    This function takes a list of Wave objects or wave file paths and
    combines them along the time dimension using xarrays built-in concatenation
    routines.

    Args:
        wave_list (list): list of wave files or Wave objects that you want to concatenate
        enhance (bool, optional): Changes variable names to something meaningful and adds attributes. Defaults to False.

    Returns:
        xarray.Dataset: waves concatenated into an xarray dataset by (time) or (time, dist)
    """
    wave_dict = {}
    for wave in wave_list:
        if not isinstance(wave, Waves):
            wave = Waves(wave)
        wave_dict[wave.file_name] = wave.to_xarray(enhance=enhance)

    ds = xr.concat(wave_dict.values(), "time")
    return ds.sortby("time")


class Waves(fileParser):
    """
    Waves Subclass.

    This class should be used when loading a CODAR wave (.wls) and WERA (.nc, .wav, .wrad_asc) wave file.
    This class utilizes the generic LLUV, WAVASC, WAVNC and WRAD classes.
    """

    def __init__(self, fname='', replace_invalid=True, grid=gpd.GeoSeries(), empty_wave=False):
        """
        Initalize a Wave object from a CTF wave file.

        Args:
            fname (str or path.Path): Filename to be loaded
            replace_invalid (bool, optional): Replace invalid values to np.nan. Defaults to True.
            grid (geopandas.GeoSeries, optional): GeoSeries containing the grid of the WERA wave file. Defaults to empty GeoSeries.
            empty_wave (bool, optional): Create an empty Wave object. Defaults to False.
        """
        logging.info("Loading wave file: {}".format(fname))

        if not fname:
            empty_wave = True
            replace_invalid = False

        super().__init__(fname)

        if self._tables:
            # Codar wave file
            if 'CTF' in self.metadata:
                if self._tables[str(1)]["data"]["DIST"].isnull().all():
                    # Averaged wave file
                    df = self._tables[str(1)]["data"]
                    self.data = df
                    self.df_index = "time"  # define index so pd.to_xarray function will automatically assign dimension and coordinates
                else:
                    # Ranged wave file
                    data_tables = []
                    for Rkey in self._tables.keys():
                        df = self._tables[Rkey]["data"]
                        data_tables.append(df)
                    self.data = pd.concat(data_tables, axis=0)
                    self.df_index = ["time", "DIST"]  # define two indices for multidimensional indexing.

                    # Remove Distance and RangeCell from metadata since they are reported for each table in _tables
                    self.metadata.pop('Distance')
                    self.metadata.pop('RangeCell')

                # Use separate date and time columns to create atetime column and drop those columns.
                self.data["time"] = self.data[["TYRS", "TMON", "TDAY", "THRS", "TMIN", "TSEC"]].apply(
                    lambda s: dt.datetime(*s), axis=1
                )
            # WERA wave file
            elif 'WAVCNC' in self._tables[str(1)]["TableType"]:
                self.data = self._tables[str(1)]["data"]
            elif 'WAVSNC' in self._tables[str(1)]["TableType"]:
                self.data = self._tables[str(1)]["data"]
            elif 'WAVCASC' in self._tables[str(1)]["TableType"]:
                self.wavc_data = self._tables[str(1)]["data"]
                self.data = self._tables[str(1)]["data"]
            elif 'WAVSASC' in self._tables[str(1)]["TableType"]:
                self.wavs_data = self._tables[str(1)]["data"]
                self.data = self._tables[str(1)]["data"]

            # Remove impossible times (before 1970 or in the future)
            epoch = dt.datetime(1970, 1, 1)
            now = dt.datetime.now()
            if not epoch <= self.time <= now:
                self.data = pd.DataFrame()

            # Remove impossible positions (only for WERA data, Codar data position is evaluated based on Distance and AntennaBearing)
            if self.is_wera:
                self.data = self.data[self.data['LOND'].between(-180, 180) & self.data['LATD'].between(-90, 90)]

        if replace_invalid:
            if 'WAVCNC' in self.metadata['FileType'] or 'WAVSNC' in self.metadata['FileType']:
                # Open the netCDF file and store it into an xarray DataSet
                wavDS=xr.open_dataset(self.full_file,decode_times=True,decode_coords='all')  
                # Get the _FillValue attribute for each variable in the dataset
                fill_values = []
                for varName in wavDS.variables:
                    var = wavDS[varName]
                    fv = var.attrs.get('_FillValue', var.encoding.get('_FillValue'))
                    if fv is not None:
                        fill_values.append(fv)
                # Reduce to the unique fill values across all variables
                values = list(set(fill_values))
                self.replace_invalid_values(list(values))
                # Drop any rows that contain NaN values for all variables (except LOND and LATD)in self.data
                self.data.dropna(subset=[col for col in self.data.columns if col not in ['LOND', 'LATD', 'GDPX', 'GDPY']], how='all', inplace=True)

            else:
                self.replace_invalid_values()

        if empty_wave:
            self.empty_wave()

        if not grid.empty:
            self.initialize_grid(grid)

        if self._iscorrupt:
            return

    def empty_wave(self):
            """
            Create an empty Wave object. The empty Wave object can be created by setting 
            the geographical grid.
            """
    
            self.file_path = ''
            self.file_name = ''
            self.full_file = ''
            self.metadata = OrderedDict()
            self._iscorrupt = False
            self.time = []
    
            for key in self._tables.keys():
                table = self._tables[key]
                self._tables[key]['TableRows'] = '0'
                if 'WAVL' in table['TableType'] or 'WAVCNC' in table['TableType'] or 'WAVSNC' in table['TableType']:
                    self.data.drop(self.data.index[:], inplace=True)
                    self._tables[key]['data'] = self.data
                elif 'WAVCASC' in table['TableType']:
                    self.wavc_data.drop(self.wavc_data.index[:], inplace=True)
                    self._tables[key]['data'] = self.wavc_data
                elif 'WAVSASC' in table['TableType']:
                    self.wavs_data.drop(self.wavs_data.index[:], inplace=True)
                    self._tables[key]['data'] = self.wavs_data
                    
            if not hasattr(self, 'data'):
                self.data = pd.DataFrame()

    def initialize_grid(self, gridGS):
        """
        Initialize the geogprahic grid for filling the LOND and LATD columns of the 
        Wave object data DataFrame.
        
        INPUT:
            gridGS: GeoPandas GeoSeries containing the longitude/latitude pairs of all
                the points in the grid
                
        OUTPUT:
            DataFrame with filled LOND and LATD columns.
        """
        
        # initialize data DataFrame with column names
        self.data = pd.DataFrame(columns=['LOND', 'LATD', 'GDPX, GDPY, MVHT, WDTO, TAVG, TNRG, QUAL'])
        
        # extract longitudes and latitude from grid GeoSeries and insert them into data DataFrame
        self.data['LOND'] = gridGS.x
        self.data['LATD'] = gridGS.y
        
        # add metadata about datum and CRS
        self.metadata = OrderedDict()
        self.metadata['GreatCircle'] = ''.join(gridGS.crs.ellipsoid.name.split()) + ' ' + str(gridGS.crs.ellipsoid.semi_major_metre) + '  ' + str(gridGS.crs.ellipsoid.inverse_flattening)

        # add attribute to indicate that this is a WERA wave file (since it has a grid)
        self.is_wera = True

    def __repr__(self):
        """
        String representation of Wave object

        Returns:
            str: string representation of Wave object
        """
        return "<Wave: {}>".format(self.file_name)

    def select_range_cell(self, rngcll=3):
            """
            This method is meant for preparing wave data as timeseries (i.e. to refer all data to a single pointfor usage as a wave buoy).
            For this reason, the method creates the field self.timeseries_data, which is a DataFrame containing only the data of the selected range cell.
            For Codar ranged wave files, select a specific range cell to be used for analysis. 
            This method filters the data to only include the specified range cell and stores the selection in self.timeseries_data.
            The method is specific for Codar wave files, it has no effect on WERA wave files.
    
            INPUT:
                rngcll (int, optional): number of the Range Cell to be selected. Defaults to 3.
            """

            if not self.is_wera:
                if len(self._tables) > 1:       # Codar ranged data
                    distance = next(
                        (d.get('Distance') for d in self._tables.values()
                        if d.get('RangeCell') == str(rngcll) and 'WAVL' in d.get('TableType', '')),
                        None
                    )

                    if distance is not None:
                        numDistance = float(distance.split()[0])
                        self.timeseries_data = self.data[self.data['DIST'] == numDistance]   
                        self.metadata['Distance'] = distance
                        self.metadata['RangeCell'] = str(rngcll)  
                else:                           # Codar averaged data
                    self.timeseries_data = self.data

    def set_reference_position(self, prefLon=None, prefLat=None, avgDistance=15000):
        """
        Set the reference latitude and longitude to refer all data to a single point(for usage as a wave buoy).
        This method is meant for preparing wave data as timeseries (i.e. to refer all data to a single pointfor usage as a wave buoy).
        For this reason, the method works on the field self.timeseries_data, which is a DataFrame containing only the data of the 
        selected range cell for Codar data or the closest grid point to the center of the grid for WERA data.
        If the latitude/longitude pair of the preferred reference position are given in input, the reference position 
        will be set to those values. Otherwise, the reference position will be calculated based on the type of wave file.
        For Codar ranged wave files, the reference position is calculated using the distance of the selected range cell and
        the antenna bearing of the HFR system.
        For Codar averaged files, the reference position is calculated using a pre-defined distance and the antenna bearing 
        of the HFR system.
        For WERA files, the reference position is set to the center of the grid and only data of the closest grid point to the reference position is kept.
        The reference longitude and latitude are stored in the metadata of the Wave object.

        INPUT:
            prefLon (float, optional): preferred longitude for the reference position. Defaults to None.
            prefLat (float, optional): preferred latitude for the reference position. Defaults to None.
            avgDistance (float, optional): average distance in meters for Codar averaged files. Defaults to 15000 m (i.e. 15 km).
        """

        # Assign the preferred reference position if given in input
        if prefLon is not None and prefLat is not None:
            if not self.is_wera:        # Codar data
                if hasattr(self, 'timeseries_data'):
                    # Add reference latitude and longitude to data
                    self.timeseries_data["LATD"] = prefLat
                    self.timeseries_data["LOND"] = prefLon
                    # Add reference latitude and longitude to metadata
                    self.metadata["ReferenceLatitude"] = str(prefLat) + ' deg'
                    self.metadata["ReferenceLongitude"] = str(prefLon) + ' deg'
                    # Create Geod object according to the Waves CRS, if defined. Otherwise use WGS84 ellipsoid
                    if 'GreatCircle' in self.metadata:
                        g = Geod(ellps=self.metadata['GreatCircle'].split()[0].replace('"',''))                  
                    else:
                        g = Geod(ellps='WGS84')
                        self.metadata['GreatCircle'] = '"WGS84"' + ' ' + str(g.a) + '  ' + str(1/g.f)
                    # Get the site latitude, longitude and antenna bearing from the metadata
                    siteLat = float(self.metadata["Origin"].split()[0])
                    siteLon = float(self.metadata["Origin"].split()[1])
                    # Evaluate distance from antenna
                    _, _, dist = g.inv(prefLon, prefLat, siteLon, siteLat)
                    # Add distance from antenna to metadata
                    self.metadata['Distance'] = f"{dist / 1000:.2f} km"
                    return
            else:                       # WERA data
                # Create Geod object according to the Waves CRS, if defined. Otherwise use WGS84 ellipsoid
                if 'GreatCircle' in self.metadata:
                    g = Geod(ellps=self.metadata['GreatCircle'].split()[0].replace('"',''))                  
                else:
                    g = Geod(ellps='WGS84')
                    self.metadata['GreatCircle'] = '"WGS84"' + ' ' + str(g.a) + '  ' + str(1/g.f)
                # Find the closest grid point to the preferred reference position
                _, _, dist = g.inv(np.full(self.data["LOND"].shape, prefLon), np.full(self.data["LATD"].shape, prefLat), self.data["LOND"], self.data["LATD"])
                idx = dist.argmin()
                closestLon = self.data["LOND"].iloc[idx]
                closestLat = self.data["LATD"].iloc[idx]
                # Keep only the data of the closest grid point
                self.timeseries_data = self.data[(self.data['LOND'] == closestLon) & (self.data['LATD'] == closestLat)]
                # Add reference latitude and longitude to metadata
                self.metadata["ReferenceLatitude"] = str(closestLat) + ' deg'
                self.metadata["ReferenceLongitude"] = str(closestLon) + ' deg'
                return
        # Otherwise, calculate the reference position based on the type of wave file
        else:
            if not self.is_wera:        # Codar data
                if hasattr(self, 'timeseries_data'):
                    # Codar wave ranged files have a distance value in the metadata while Codar wave averaged files do not.
                    if 'Distance' in self.metadata:
                        numDistance = float(self.metadata['Distance'].split()[0])
                        if 'km' in self.metadata['Distance']:
                            numDistance *= 1000  # convert km to meters
                    else:
                        # For Codar wave averaged files, a pre-defined distance value is used.
                        numDistance = avgDistance
                        self.metadata['Distance'] = f"{numDistance / 1000:.2f} km"

                    # Get the site latitude, longitude and antenna bearing from the metadata
                    siteLat = float(self.metadata["Origin"].split()[0])
                    siteLon = float(self.metadata["Origin"].split()[1])
                    siteBearing = float(self.metadata["AntennaBearing"].split()[0])
                    # Create Geod object according to the Total CRS, if defined. Otherwise use WGS84 ellipsoid
                    if 'GreatCircle' in self.metadata:
                        g = Geod(ellps=self.metadata['GreatCircle'].split()[0].replace('"',''))                  
                    else:
                        g = Geod(ellps='WGS84')
                        self.metadata['GreatCircle'] = '"WGS84"' + ' ' + str(g.a) + '  ' + str(1/g.f)
                    # Calculate the reference latitude and longitude using the Geod object
                    refLon, refLat, back_azimuth = g.fwd(siteLon, siteLat, siteBearing, numDistance)                
                    # Add reference latitude and longitude to data
                    self.timeseries_data["LATD"] = refLat
                    self.timeseries_data["LOND"] = refLon
                    # Add reference latitude and longitude to metadata
                    self.metadata["ReferenceLatitude"] = str(refLat) + ' deg'
                    self.metadata["ReferenceLongitude"] = str(refLon) + ' deg'
                    return
            else:                       # WERA data
                # Get longitude limits   
                if 'BBminLongitude' in self.metadata:
                    lonMin = float(self.metadata['BBminLongitude'].split()[0])
                else:
                    lonMin = self.data.LOND.min()
                if 'BBmaxLongitude' in self.metadata:
                    lonMax = float(self.metadata['BBmaxLongitude'].split()[0])
                else:
                    lonMax = self.data.LOND.max()
                # Get latitude limits   
                if 'BBminLatitude' in self.metadata:
                    latMin = float(self.metadata['BBminLatitude'].split()[0])
                else:
                    latMin = self.data.LATD.min()
                if 'BBmaxLatitude' in self.metadata:
                    latMax = float(self.metadata['BBmaxLatitude'].split()[0])
                else:
                    latMax = self.data.LATD.max()   
                # Evaluate the center of the grid as the reference position
                centerLon = (lonMin + lonMax) / 2
                centerLat = (latMin + latMax) / 2
                # Create Geod object according to the Total CRS, if defined. Otherwise use WGS84 ellipsoid
                if 'GreatCircle' in self.metadata:
                    g = Geod(ellps=self.metadata['GreatCircle'].split()[0].replace('"',''))                  
                else:
                    g = Geod(ellps='WGS84')
                    self.metadata['GreatCircle'] = '"WGS84"' + ' ' + str(g.a) + '  ' + str(1/g.f)
                # Find the closest grid point to the center grid position
                _, _, dist = g.inv(np.full(self.data["LOND"].shape, centerLon), np.full(self.data["LATD"].shape, centerLat), self.data["LOND"], self.data["LATD"])
                idx = dist.argmin()
                closestLon = self.data["LOND"].iloc[idx]
                closestLat = self.data["LATD"].iloc[idx]
                # Keep only the data of the closest grid point
                self.timeseries_data = self.data[(self.data['LOND'] == closestLon) & (self.data['LATD'] == closestLat)]
                # Add reference latitude and longitude to metadata
                self.metadata["ReferenceLatitude"] = str(closestLat) + ' deg'
                self.metadata["ReferenceLongitude"] = str(closestLon) + ' deg'
                return

    def to_xarray_timeseries(self):
            """
            This function creates a dictionary of xarray DataArrays containing the variables
            of the Waves object bidimensionally expanded along the coordinate axes (T,Z).  
            The coordinate axes are set as (TIME, DEPTH) in order to represent the variables 
            of the Waves object as a time-series. LATITUDE and LONGITUDE are set as separate,
            time-independent DataArrays.
            The LATITUDE and LONGITUDE values are taken from the Wave object timeseries DataFrame.
            The generated dictionary is attached to the Total object, named as xts.
    
            """
            if hasattr(self, 'timeseries_data'):
                # Initialize empty dictionary
                xts = OrderedDict()

                # Process Codar data
                if not self.is_wera:
                    # Sort time values
                    df = self.timeseries_data.sort_values('time')

                    # Evaluate timestamp as number of days since 1950-01-01T00:00:00Z
                    timeDelta = df['time'].values - np.datetime64("1950-01-01T00:00:00")
                    df['time'] = timeDelta / np.timedelta64(1, "D")

                    # Set coordinate axes
                    time = (df['time'].values).astype(np.float64)                      # TIME axis (length N)
                    depth = np.array([0], dtype=float)                                  # DEPTH axis (length 1)
                    coords = {"TIME": ("TIME", time), "DEPTH": ("DEPTH", depth)}

                    # Set the columns that become (TIME, DEPTH) variables
                    variable_cols = [c for c in df.columns if c not in ('TIME', 'time', 'LOND', 'LATD', 'TYRS', 'TMON', 'TDAY', 'THRS', 'TMIN', 'TSEC')]
                else:
                    # Evaluate timestamp as number of days since 1950-01-01T00:00:00Z
                    timeDelta = self.time - dt.datetime.strptime('1950-01-01T00:00:00Z','%Y-%m-%dT%H:%M:%SZ')
                    ncTime = timeDelta.days + timeDelta.seconds / (60*60*24)

                    # Set coordinate axes
                    time = np.array([ncTime], dtype=np.float64)                 # TIME axis (length 1)
                    depth = np.array([0], dtype=float)                          # DEPTH axis (length 1)
                    coords = {"TIME": ("TIME", time), "DEPTH": ("DEPTH", depth)}

                    # Set the columns that become (TIME, DEPTH) variables
                    df = self.timeseries_data
                    variable_cols = [c for c in df.columns if c not in ('LOND', 'LATD')]

                # Add DataArray for data variables
                for col in variable_cols:
                    data = df[col].values[:, np.newaxis]            # (N,) -> (N, 1)
                    xts[col] = xr.DataArray(
                        data=data,
                        dims=("TIME", "DEPTH"),
                        coords=coords,
                        name=col,
                    )

                data = df['LOND'].values[:,np.newaxis]
                xts['LONGITUDE'] = xr.DataArray(
                                        data=data,
                                        name='LONGITUDE',
                                    )
                data = df['LATD'].values[:,np.newaxis]
                xts['LATITUDE'] = xr.DataArray(
                                        data=data,
                                        name='LATITUDE',
                                    )

                # Add DataArray for coordinate variables
                xts['TIME'] = xr.DataArray(time,
                                        dims={'TIME': len(time)},
                                        coords={'TIME': len(time)})
                xts['DEPTH'] = xr.DataArray(0,
                                        dims={'DEPTH': 1},
                                        coords={'DEPTH': [0]})
                # xts['LATITUDE'] = xr.DataArray(df['LATD'].iloc[0],
                #                             dims={'LATITUDE': df['LATD'].iloc[0]},
                #                             coords={'LATITUDE': df['LATD'].iloc[0]})
                # xts['LONGITUDE'] = xr.DataArray(df['LOND'].iloc[0],
                #                                 dims={'LONGITUDE': df['LOND'].iloc[0]},
                #                                 coords={'LONGITUDE': df['LOND'].iloc[0]})  
                
                # Attach the dictionary to the Total object
                self.xts = xts
                
                return

    def check_ehn_mandatory_variables(self):
        """
        This function checks if the Waves object contains all the mandatory data variables
        (i.e. not coordinate variables) required by the European standard data model developed in the framework of the 
        EuroGOOS HFR Task Team.
        Missing variables are appended to the DataFrame containing data, filled with NaNs.
        
        INPUT:            
            
        OUTPUT:
        """
        # Set mandatory variables based on the HFR manufacturer
        if self.is_wera:
            chkVars = ['MWHT', 'WDTO', 'TAVG', 'QUAL']
        else:
            chkVars = ['MWHT', 'WAVB']
            
        # Check variables and add missing ones
        for vv in chkVars:
            if vv not in self.data.columns:
                self.data[vv] = np.nan
                
        return

    def apply_instac_datamodel(self, network_data, station_data, version):
        """
        This function applies the Copernicus Marine Service In Situ TAC data model 
        to the Waves object.
        The Waves object content is stored into an xarray Dataset built from the
        xarray DataArrays created by the Waves method to_xarray_timeseries.
        Variable data types information are collected from
        "Data_Models/CMEMS_IN_SITU_TAC/Waves/Waves_Data_Packing.json" file.
        Variable attribute schema is collected from 
        "Data_Models/CMEMS_IN_SITU_TAC/Waves/Waves_Variables.json" file.
        Global attribute schema is collected from 
        "Data_Models/CMEMS_IN_SITU_TAC/Waves/Waves_Global_Attributes.json" file.
        Global attributes are created starting from Waves object metadata and from 
        DataFrames containing the information about HFR network and radial stations
        read from the EU HFR NODE database.
        The generated xarray Dataset is attached to the Waves object, named as xds_TS.
        
        INPUT:
            network_data: DataFrame containing the information of the network to which the radial site belongs
            station_data: DataFrame containing the information of the radial site that produced the radial
            version: version of the data model
            
            
        OUTPUT:
        """
        # Set the netCDF format
        ncFormat = 'NETCDF4_CLASSIC'
        
        # Get bounding box limits and grid resolution from database
        lonMin = network_data.iloc[0]['geospatial_lon_min']
        lonMax = network_data.iloc[0]['geospatial_lon_max']
        latMin = network_data.iloc[0]['geospatial_lat_min']
        latMax = network_data.iloc[0]['geospatial_lat_max']
        gridRes = network_data.iloc[0]['grid_resolution']*1000

        # Expand Waves object variables along the coordinate axes
        self.to_xarray_timeseries()
        
        # Set auxiliary coordinate sizes
        maxsiteSize = 150
        refmaxSize = 50
        maxinstSize = 50
        
        # Get data packing information per variable
        f = open('Data_Models/CMEMS_IN_SITU_TAC/Waves/Waves_Data_Packing.json')
        dataPacking = json.loads(f.read())
        f.close()
        
        # Get variable attributes
        f = open('Data_Models/CMEMS_IN_SITU_TAC/Waves/Waves_Variables.json')
        wavVariables = json.loads(f.read())
        f.close()
        
        # Get global attributes
        f = open('Data_Models/CMEMS_IN_SITU_TAC/Waves/Waves_Global_Attributes.json')
        globalAttributes = json.loads(f.read())
        f.close()
        
        # Rename significatn wave height, wave period and wave direction variables
        self.xts['VHM0'] = self.xts.pop('MWHT')
        if 'TAVG' in self.xts:                          # Only for WERA files
            self.xts['VM01'] = self.xts.pop('TAVG')
        if 'WAVB' in self.xts:                          # Wave direction from (Codar)
            self.xts['VMDR'] = self.xts.pop('WAVB')
        if 'WDTO' in self.xts:                          # Wave direction to (WERA) -> to be converted into direction from
            self.xts['VMDR'] = self.xts.pop('WDTO')
            self.xts["VMDR"] = (self.xts["VMDR"] + 180) % 360
        
        # Drop unnecessary DataArrays from the DataSet
        toDrop = ['GDPX', 'GDPY', 'TNRG', 'QUAL', 'MWPD','WNDB', 'PMWH', 'ACNT', 'DIST','RCLL', 'WDPT', 'MTHD', 'FLAG', 'WHNM', 'WHSD', 'TYRS', 'TMON', 'TDAY', 'THRS', 'TMIN', 'TSEC', 'time', 'LATD', 'LOND']
        for t in toDrop:
            if t in self.xts:
                self.xts.pop(t)
        toDrop = []
        for vv in self.xts:
            if vv not in wavVariables.keys():
                toDrop.append(vv)
        for rv in toDrop:
            self.xts.pop(rv)            
            
        # Add coordinate reference system to the dictionary
        self.xts['crs'] = xr.DataArray(int(0), )       
        
        # Add antenna related variables to the dictionary
        # Number of antennas        
        contributingSiteNrx = station_data.loc[station_data['station_id']]['number_of_receive_antennas'].to_numpy()
        nRX = np.asfarray(contributingSiteNrx)
        nRX = np.pad(nRX, (0, maxsiteSize - len(nRX)), 'constant',constant_values=(np.nan,np.nan))
        contributingSiteNtx = station_data.loc[station_data['station_id']]['number_of_transmit_antennas'].to_numpy()
        nTX = np.asfarray(contributingSiteNtx)
        nTX = np.pad(nTX, (0, maxsiteSize - len(nTX)), 'constant',constant_values=(np.nan,np.nan))
        self.xts['NARX'] = xr.DataArray([nRX], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xts['NATX'] = xr.DataArray([nTX], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        
        # Longitude and latitude of antennas
        contributingSiteLat = station_data.loc[station_data['station_id']]['site_lat'].to_numpy()
        siteLat = np.pad(contributingSiteLat, (0, maxsiteSize - len(contributingSiteLat)), 'constant',constant_values=(np.nan,np.nan))
        contributingSiteLon = station_data.loc[station_data['station_id']]['site_lon'].to_numpy()
        siteLon = np.pad(contributingSiteLon, (0, maxsiteSize - len(contributingSiteLon)), 'constant',constant_values=(np.nan,np.nan))
        self.xts['SLTR'] = xr.DataArray([siteLat], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xts['SLNR'] = xr.DataArray([siteLon], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xts['SLTT'] = xr.DataArray([siteLat], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xts['SLNT'] = xr.DataArray([siteLon], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        
        # Codes of antennas
        contributingSiteCodeList = station_data.loc[station_data['station_id']]['station_id'].tolist()
        antCode = np.array([site.encode() for site in contributingSiteCodeList])
        antCode = np.pad(antCode, (0, maxsiteSize - len(contributingSiteCodeList)), 'constant',constant_values=('',''))
        self.xts['SCDR'] = xr.DataArray(np.array([antCode]), dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xts['SCDR'].encoding['char_dim_name'] = 'STRING' + str(len(station_data['station_id'].to_numpy()[0]))
        self.xts['SCDT'] = xr.DataArray(np.array([antCode]), dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xts['SCDT'].encoding['char_dim_name'] = 'STRING' + str(len(station_data['station_id'].to_numpy()[0]))
                
        # Add SDN namespace variables to the dictionary
        siteCode = ('%s' % network_data.iloc[0]['network_id']).encode()
        self.xts['SDN_CRUISE'] = xr.DataArray([siteCode], dims={'TIME': len(pd.date_range(self.time, periods=1))})
        self.xts['SDN_CRUISE'].encoding['char_dim_name'] = 'STRING' + str(len(siteCode))
        platformCode = ('%s' % network_data.iloc[0]['network_id'] + '-Total').encode()
        self.xts['SDN_STATION'] = xr.DataArray([platformCode], dims={'TIME': len(pd.date_range(self.time, periods=1))})
        self.xts['SDN_STATION'].encoding['char_dim_name'] = 'STRING' + str(len(platformCode))
        ID = ('%s' % platformCode.decode() + '_' + self.time.strftime('%Y-%m-%dT%H:%M:%SZ')).encode()
        self.xts['SDN_LOCAL_CDI_ID'] = xr.DataArray([ID], dims={'TIME': len(pd.date_range(self.time, periods=1))})
        self.xts['SDN_LOCAL_CDI_ID'].encoding['char_dim_name'] = 'STRING' + str(len(ID))
        sdnEDMO = np.asfarray(pd.concat([network_data['EDMO_code'],station_data['EDMO_code']]).unique())
        sdnEDMO = np.pad(sdnEDMO, (0, maxinstSize - len(sdnEDMO)), 'constant',constant_values=(np.nan,np.nan))
        self.xts['SDN_EDMO_CODE'] = xr.DataArray([sdnEDMO], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXINST': maxinstSize})
        sdnRef = ('%s' % network_data.iloc[0]['metadata_page']).encode()
        self.xts['SDN_REFERENCES'] = xr.DataArray([sdnRef], dims={'TIME': len(pd.date_range(self.time, periods=1))})
        self.xts['SDN_REFERENCES'].encoding['char_dim_name'] = 'STRING' + str(len(sdnRef))
        sdnXlink = ('%s' % '<sdn_reference xlink:href=\"' + sdnRef.decode() + '\" xlink:role=\"\" xlink:type=\"URL\"/>').encode()
        self.xts['SDN_XLINK'] = xr.DataArray(np.array([[sdnXlink]]), dims={'TIME': len(pd.date_range(self.time, periods=1)), 'REFMAX': refmaxSize})
        self.xts['SDN_XLINK'].encoding['char_dim_name'] = 'STRING' + str(len(sdnXlink))
        
        # Add spatial and temporal coordinate QC variables (set to good data due to the nature of HFR system)
        self.xts['TIME_QC'] = xr.DataArray([1],dims={'TIME': len(pd.date_range(self.time, periods=1))})
        self.xts['POSITION_QC'] = self.xts['OWTR_QC']
        self.xts['DEPTH_QC'] = xr.DataArray([1],dims={'TIME': len(pd.date_range(self.time, periods=1))})
            
        # Create DataSet from DataArrays
        self.xds = xr.Dataset(self.xts)
        
        # Add data variable attributes to the DataSet
        for vv in self.xds:
            self.xds[vv].attrs = wavVariables[vv]
            
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
            self.xds[cc].attrs = wavVariables[cc]
            
        # Evaluate measurement maximum depth
        vertMax = 3e8 / (8*np.pi * station_data['transmit_central_frequency'].to_numpy().min()*1e6)
        
        # Evaluate time coverage start, end, resolution and duration
        timeCoverageStart = self.time - relativedelta(minutes=network_data.iloc[0]['temporal_resolution']/2)
        timeCoverageEnd = self.time + relativedelta(minutes=network_data.iloc[0]['temporal_resolution']/2)
        timeResRD = relativedelta(minutes=network_data.iloc[0]['temporal_resolution'])
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
        globalAttributes.pop('oceanops_ref')
        globalAttributes.pop('wmo_platform_code')
        globalAttributes.pop('wigos_id')
        globalAttributes['doa_estimation_method'] = ', '.join(station_data[["station_id", "DoA_estimation_method"]].apply(": ".join, axis=1))
        globalAttributes['calibration_type'] = ', '.join(station_data[["station_id", "calibration_type"]].apply(": ".join, axis=1))
        if 'HFR-US' in network_data.iloc[0]['network_id']:
            station_data['last_calibration_date'] = 'N/A'
            globalAttributes['last_calibration_date'] = ', '.join(pd.concat([station_data['station_id'],station_data['last_calibration_date']],axis=1)[["station_id", "last_calibration_date"]].apply(": ".join, axis=1))
        else:
            globalAttributes['last_calibration_date'] = ', '.join(pd.concat([station_data['station_id'],station_data['last_calibration_date'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%SZ'))],axis=1)[["station_id", "last_calibration_date"]].apply(": ".join, axis=1))
            globalAttributes['last_calibration_date'] = globalAttributes['last_calibration_date'].replace('1-01-01T00:00:00Z', 'N/A')
        globalAttributes['calibration_link'] = ', '.join(station_data[["station_id", "calibration_link"]].apply(": ".join, axis=1))
        # globalAttributes['title'] = network_data.iloc[0]['title']
        globalAttributes['title'] = 'Near Real Time Ocean Wave parameters by ' + globalAttributes['platform_code']
        globalAttributes['summary'] = network_data.iloc[0]['summary']
        globalAttributes['institution'] = ', '.join(pd.concat([network_data['institution_name'],station_data['institution_name']]).unique().tolist())
        globalAttributes['institution_edmo_code'] = ', '.join([str(x) for x in pd.concat([network_data['EDMO_code'],station_data['EDMO_code']]).unique().tolist()])
        globalAttributes['institution_references'] = ', '.join(pd.concat([network_data['institution_website'],station_data['institution_website']]).unique().tolist())
        globalAttributes['id'] = ID.decode()
        globalAttributes['project'] = network_data.iloc[0]['project']
        globalAttributes['comment'] = network_data.iloc[0]['comment']
        globalAttributes['network'] = network_data.iloc[0]['network_name']
        globalAttributes['data_type'] = globalAttributes['data_type'].replace('current data', 'total current data')
        globalAttributes['geospatial_lat_min'] = str(network_data.iloc[0]['geospatial_lat_min'])
        globalAttributes['geospatial_lat_max'] = str(network_data.iloc[0]['geospatial_lat_max'])
        globalAttributes['geospatial_lat_resolution'] = str(network_data.iloc[0]['grid_resolution'])
        globalAttributes['geospatial_lon_min'] = str(network_data.iloc[0]['geospatial_lon_min'])
        globalAttributes['geospatial_lon_max'] = str(network_data.iloc[0]['geospatial_lon_max'])
        globalAttributes['geospatial_lon_resolution'] = str(network_data.iloc[0]['grid_resolution'])        
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
        globalAttributes['processing_level'] = '3B'
        globalAttributes['contributor_name'] = network_data.iloc[0]['contributor_name']
        globalAttributes['contributor_role'] = network_data.iloc[0]['contributor_role']
        globalAttributes['contributor_email'] = network_data.iloc[0]['contributor_email']
        globalAttributes['manufacturer'] = ', '.join(station_data[["station_id", "manufacturer"]].apply(": ".join, axis=1))
        globalAttributes['sensor_model'] = ', '.join(station_data[["station_id", "manufacturer"]].apply(": ".join, axis=1))
        globalAttributes['software_version'] = version
        
        creationDate = dt.datetime.now(timezone.utc)
        globalAttributes['metadata_date_stamp'] = creationDate.strftime('%Y-%m-%dT%H:%M:%SZ')
        globalAttributes['date_created'] = creationDate.strftime('%Y-%m-%dT%H:%M:%SZ')
        globalAttributes['date_modified'] = creationDate.strftime('%Y-%m-%dT%H:%M:%SZ')
        globalAttributes['history'] = 'Data collected at ' + self.time.strftime('%Y-%m-%dT%H:%M:%SZ') + '. netCDF file created at ' \
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
            if 'valid_min' in wavVariables[vv]:
                if ('scale_factor' in dataPacking[vv]) and ('add_offset' in dataPacking[vv]):
                    self.xds[vv].attrs['valid_min'] = np.float_(((wavVariables[vv]['valid_min'] - dataPacking[vv]['add_offset']) / dataPacking[vv]['scale_factor'])).astype(dataPacking[vv]['dtype'])
                else:
                    self.xds[vv].attrs['valid_min'] = np.float_(wavVariables[vv]['valid_min']).astype(dataPacking[vv]['dtype'])
            if 'valid_max' in wavVariables[vv]:             
                if ('scale_factor' in dataPacking[vv]) and ('add_offset' in dataPacking[vv]):
                    self.xds[vv].attrs['valid_max'] = np.float_(((wavVariables[vv]['valid_max'] - dataPacking[vv]['add_offset']) / dataPacking[vv]['scale_factor'])).astype(dataPacking[vv]['dtype'])
                else:
                    self.xds[vv].attrs['valid_max'] = np.float_(wavVariables[vv]['valid_max']).astype(dataPacking[vv]['dtype'])
            
        # Encode data types and avoid data packing, valid_min, valid_max and _FillValue for the coordinate variables of the DataSet
        for cc in self.xds.coords:
            if cc in dataPacking:
                if 'dtype' in dataPacking[cc]:
                    self.xds[cc].encoding['dtype'] = dataPacking[cc]['dtype']
                if 'valid_min' in wavVariables[cc]:
                    del self.xds[cc].attrs['valid_min']
                if 'valid_max' in wavVariables[cc]:
                    del self.xds[cc].attrs['valid_max']
                self.xds[cc].encoding['_FillValue'] = None
                
        return
    
    def mask_over_land(self, timeseries=False, subset=False, res='high'):
        """
        This function masks the wave data lying on land.        
        Wave data coordinates are checked against a reference file containing information 
        about which locations are over land or in an unmeasurable area (for example, behind an 
        island or point of land). 
        The Natural Earth public domain maps are used as reference.
        If "res" option is set to "high", the map with 10 m resolution is used, otherwise the map with 110 m resolution is used.
        The EPSG:4326 CRS is used for distance calculations.
        If "subset" option is set to True, the wave data lying on land are removed.
        Based on the input option "timeseries", the method is applied either to all data entries or only to the timeseries data.
        
        INPUT:
            timeseries: option enabling the application of the method to the timeseries_data DataFrame (if set to True) or
                        to the data DataFrame (if set to False). Defaults to False.
            subset: option enabling the removal of wave data on land (if set to True)
            res: resolution of the www.naturalearthdata.com dataset used to perform the masking; None or 'low' or 'high'. Defaults to 'high'.
            
        OUTPUT:
            waterIndex: list containing the indices of wave data lying on water.
        """
        # Check the DataFrame to be masked
        if timeseries:
            if hasattr(self, 'timeseries_data'):
                df = self.timeseries_data
            else:
                return
        else:
            df = self.data

        # Check if the selected DataFrame has LOND and LATD columns
        if 'LOND' in df.columns and 'LATD' in df.columns:        
            # Load the reference file (GeoPandas "naturalearth_lowres")
            mask_dir = '.hfradarpy'
            if (res == 'high'):
                maskfile = os.path.join(mask_dir, 'ne_10m_admin_0_countries.shp')
            else:
                maskfile = os.path.join(mask_dir, 'ne_110m_admin_0_countries.shp')
            land = gpd.read_file(maskfile)

            # Build the GeoDataFrame containing wave position points
            geodata = gpd.GeoDataFrame(
                df[['LOND', 'LATD']],
                crs="EPSG:4326",
                geometry=[
                    Point(xy) for xy in zip(df.LOND.values, df.LATD.values)
                ]
            )
            # Join the GeoDataFrame containing wave position points with GeoDataFrame containing leasing areas
            geodata = gpd.sjoin(geodata.to_crs(4326), land.to_crs(4326), how="left", predicate="intersects")

            # All data in the continent column that lies over water should be nan.
            waterIndex = geodata['CONTINENT'].isna()

            if subset:
                # Subset the data to water only
                if timeseries:
                    self.timeseries_data = self.timeseries_data.loc[waterIndex].reset_index()
                else:
                    self.data = self.data.loc[waterIndex].reset_index()
            else:
                return waterIndex

    def initialize_qc(self):
        """
        Initialize dictionary entry for QC metadata.
        """
        # Initialize dictionary entry for QC metadta
        self.metadata['QCTest'] = {}

    def qc_ehn_over_water(self):
        """
        This test labels wave data that lie on water with a good data” flag.
        Otherwise the wave data are labeled with a “bad data” flag.
        The ARGO QC flagging scale is used.
        
        Wave data coordinates are checked against a reference file containing information 
        about which locations are over land or in an unmeasurable area (for example, behind an 
        island or point of land). 
        The Natural Earth public domain maps are used as reference.
        
        This test was defined in the framework of the EuroGOOS HFR Task Team.
        """
        # Set the test name
        testName = 'OWTR_QC'
        
        # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
        if 'LOND' in self.data.columns and 'LATD' in self.data.columns:
            self.data.loc[:,testName] = 1
        if hasattr(self, 'timeseries_data'):
            self.timeseries_data.loc[:,testName] = 1
        
        # Set bad flag where land is flagged (mask_over_land method)
        if 'LOND' in self.data.columns and 'LATD' in self.data.columns: 
            self.data.loc[~self.mask_over_land(subset=False), testName] = 4
        if hasattr(self, 'timeseries_data'):
            self.timeseries_data.loc[~self.mask_over_land(timeseries=True, subset=False), testName] = 4
            
        self.metadata['QCTest'][testName] = 'Over Water QC Test - Thresholds=[' + 'GeoPandas "naturalearth_lowres"]'
        
        
    def qc_instac_global_range(self):
        """
        This test applies a gross filter on observed values for waves. It needs to accommodate all the expected extremes encountered in the oceans.
        The applied ranges are:
        - Significant and mean wave height in range 0m to 25m.
        - Mean wave period in range 1s to 25s.
        - Peak period in range 1s to 30s.
        - Wave directions and angular spreading in range 0º to 360º.
        This test applies to wave data as timeseries (i.e. all data are referred to a single point
        for usage as a wave buoy). Thus, the method works on the field self.timeseries_data.
        For each timestamp and position, if all the interested variables have values falling within the specified
        ranges, the data is labeled with a "good data" flag.
        Otherwise the data is labeled with a “bad data” flag.
        The ARGO QC flagging scale is used.
        
        This test was defined in the framework of the Copernicus Marine Service In Situ TAC and described
        in Copernicus In Situ TAC, Real Time Quality Control for WAVES, https://doi.org/10.13155/46607
        
        """
        # Set the test name
        testName = 'GRNG_QC'

        # Set the range limits for the data variables
        HsLim = 25      # meters
        TmnLim = 25     # seconds
        TpkLim = 30     # seconds
        
        # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
        self.timeseries_data.loc[:,testName] = 1

        # set bad flag for significant wave heights out of range
        self.timeseries_data.loc[(self.timeseries_data['MWHT'] < 0), testName] = 4
        self.timeseries_data.loc[(self.timeseries_data['MWHT'] > HsLim), testName] = 4
        self.timeseries_data.loc[self.timeseries_data["MWHT"].isna(), testName] = np.nan
        
        # set bad flag for wave period out of range
        if self.is_wera:
            self.timeseries_data.loc[(self.timeseries_data['TAVG'] < 1), testName] = 4
            self.timeseries_data.loc[(self.timeseries_data['TAVG'] > TmnLim), testName] = 4
            self.timeseries_data.loc[self.timeseries_data["TAVG"].isna(), testName] = np.nan

        # set bad flag for wave direction out of range
        if self.is_wera:
            self.timeseries_data.loc[(self.timeseries_data['WDTO'] < 0), testName] = 4
            self.timeseries_data.loc[(self.timeseries_data['WDTO'] > 360), testName] = 4
            self.timeseries_data.loc[self.timeseries_data["WDTO"].isna(), testName] = np.nan
        else:
            self.timeseries_data.loc[(self.timeseries_data['WAVB'] < 0), testName] = 4
            self.timeseries_data.loc[(self.timeseries_data['WAVB'] > 360), testName] = 4
            self.timeseries_data.loc[self.timeseries_data["WAVB"].isna(), testName] = np.nan
        
        self.metadata['QCTest'][testName] = 'Global Range QC Test - Test applies to each timestamp and position. ' \
            + 'Thresholds=[' + f'Significant wave height limit={str(HsLim)} (m) ' + f'Mean wave period limit={str(TmnLim)} (s)]'

    def clean_header(self):
        """
        Cleans the header data from the wave data for proper input into MySQL database
        """

        keep = [
            "TimeCoverage",
            "WaveMinDopplerPoints",
            "AntennaBearing",
            "DopplerCells",
            "TransmitCenterFreqMHz",
            "CTF",
            "TableColumnTypes",
            "TimeZone",
            "WaveBraggPeakDropOff",
            "RangeResolutionKMeters",
            "CoastlineSector",
            "WaveMergeMethod",
            "RangeCells",
            "WaveBraggPeakNull",
            "WaveUseInnerBragg",
            "BraggSmoothingPoints",
            "Manufacturer",
            "TimeStamp",
            "FileType",
            "TableRows",
            "BraggHasSecondOrder",
            "Origin",
            "MaximumWavePeriod",
            "UUID",
            "WaveBraggNoiseThreshold",
            "TransmitBandwidthKHz",
            "Site",
            "TransmitSweepRateHz",
            "WaveBearingLimits",
            "WavesFollowTheWind",
            "CurrentVelocityLimit",
        ]

        key_list = list(self.metadata.keys())
        for key in key_list:
            if key not in keep:
                del self.metadata[key]

        for k, v in self.metadata.items():
            if "Site" in k:
                self.metadata[k] = "".join(e for e in v if e.isalnum())
            elif "TimeStamp" in k:
                t_list = v.split()
                t_list = [int(s) for s in t_list]
                self.metadata[k] = dt.datetime(t_list[0], t_list[1], t_list[2], t_list[3], t_list[4], t_list[5]).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            elif k in ("TimeCoverage", "RangeResolutionKMeters"):
                self.metadata[k] = re.findall(r"\d+\.\d+", v)[0]
            elif k in ("WaveMergeMethod", "WaveUseInnerBragg", "WavesFollowTheWind"):
                self.metadata[k] = re.search(r"\d+", v).group()
            elif "TimeZone" in k:
                self.metadata[k] = re.search('"(.*)"', v).group(1)
            elif k in ("WaveBearingLimits", "CoastlineSector"):
                bearings = re.findall(r"[-+]?\d*\.\d+|\d+", v)
                self.metadata[k] = ", ".join(e for e in bearings)
            else:
                continue

    def flag_wave_heights(self, minimum=0.2, maximum=5, remove=False):
        """
        Flag bad wave heights in Wave instance. This method labels wave heights between wave_min and wave_max as good,
        while labeling anything else bad

        Args:
            minimum (float, optional): Minimum Wave Height - Waves above this will be considered good. Defaults to 0.2.
            maximum (int, optional): Maximum Wave Height - Waves less than this will be considered good. Defaults to 5.
            remove (bool, optional): Remove bad wave heights. Defaults to False.
        """
        boolean = self.data["MWHT"].between(minimum, maximum, inclusive="both")

        if not remove:
            self.data["mwht_flag"] = 1
            self.data["mwht_flag"] = self.data["mwht_flag"].where(boolean, other=4)
        elif remove:
            self.data = self.data[boolean]

    def to_xarray(self, enhance=False):
        """
        Convert Wave data from a Pandas DataFrame to an xarray Dataset.

        Args:
            enhance (bool, optional): Rename variables to something meaningful and add useful attributes. Defaults to False.

        Returns:
            xarray.Dataset: xarray dataset containing converted wave data.
        """
        logging.info("Converting wave data to xarray dataset")

        # Set dataframe to indexes defined during class initialization
        tdf = self.data.set_index(self.df_index).drop(["TIME", "TYRS", "TMON", "TDAY", "THRS", "TMIN", "TSEC"], axis=1)

        # Intitialize xarray dataset
        ds = tdf.to_xarray()

        # Assign header data to global attributes
        ds = ds.assign_attrs(self.metadata)

        if enhance is True:
            # global_attr = required_global_attributes(required_attributes, time_start, time_end)
            ds = self.enhance_xarray(ds)
            ds = xr.decode_cf(ds)

        return ds

    def enhance_xarray(self, xds):
        """
        Rename variables to meaningful names.
        Add attributes to help the dataset be self-describing.

        Args:
            xds (xarray.Dataset): xarray.Dataset containing wave CTF files

        Returns:
            xarray.Dataset: enhanced wave file xarray.Dataset
        """
        rename = dict()
        rename["MWHT"] = "wave_height"
        rename["MWPD"] = "wave_period"
        rename["WAVB"] = "wave_bearing"
        rename["WNDB"] = "wind_bearing"
        rename["ACNT"] = "cross_spectra_averaged_count"
        rename["DIST"] = "distance_from_origin"
        rename["RCLL"] = "range_cell_result"
        rename["WDPT"] = "doppler_points_used"
        rename["MTHD"] = "wave_method"
        rename["FLAG"] = "vector_flag"

        if "PMWH" in self.data.keys():
            rename["PMWH"] = "maximum_observable_wave_height"

        if "WHNM" in self.data.keys():
            rename["WHNM"] = "num_valid_source_wave_vectors"

        if "WHSD" in self.data.keys():
            rename["WHSD"] = "standard_deviation_of_wave_heights"

        # rename variables to something meaningful if they existin
        # in the xarray dataset
        existing_renames = {k: v for k, v in rename.items() if k in xds}
        xds = xds.rename(existing_renames)

        length = len(xds.time)
        lonlat = [float(x) for x in self.metadata["Origin"].split()]
        xds["lon"] = xr.DataArray(np.full(length, lonlat[1]), dims=("time"))
        xds["lat"] = xr.DataArray(np.full(length, lonlat[0]), dims=("time"))

        # set time attribute
        xds["time"].attrs["standard_name"] = "time"
        xds["time"].attrs["long_name"] = "Universal Time Coordinated (UTC) Time"

        # Set wave_height attributes
        xds["wave_height"].attrs["long_name"] = "wave model height in meters"
        xds["wave_height"].attrs["standard_name"] = "sea_surface_wave_significant_height"
        xds["wave_height"].attrs["units"] = "m"
        xds["wave_height"].attrs["comment"] = "wave model height in meters for every one of three waves"
        xds["wave_height"].attrs["valid_min"] = np.double(0)
        xds["wave_height"].attrs["valid_max"] = np.double(100)
        xds["wave_height"].attrs["coordinates"] = "time"
        xds["wave_height"].attrs["grid_mapping"] = "crs"
        xds["wave_height"].attrs["coverage_content_type"] = "physicalMeasurement"

        # Set wave_period attributes
        xds["wave_period"].attrs["long_name"] = "wave spectra period in seconds"
        xds["wave_period"].attrs["standard_name"] = "sea_surface_wave_mean_period"
        xds["wave_period"].attrs["units"] = "s"
        xds["wave_period"].attrs["comment"] = "wave spectra period in seconds"
        xds["wave_period"].attrs["valid_min"] = np.double(0)
        xds["wave_period"].attrs["valid_max"] = np.double(100)
        xds["wave_period"].attrs["coordinates"] = "time"
        xds["wave_period"].attrs["grid_mapping"] = "crs"
        xds["wave_period"].attrs["coverage_content_type"] = "physicalMeasurement"

        # Set wave_bearing attributes
        xds["wave_bearing"].attrs["long_name"] = "wave from direction in degrees"
        xds["wave_bearing"].attrs["standard_name"] = "sea_surface_wave_from_direction"
        xds["wave_bearing"].attrs["units"] = "degrees"
        xds["wave_bearing"].attrs["comment"] = "wave from direction in degrees"
        xds["wave_bearing"].attrs["valid_min"] = np.double(0)
        xds["wave_bearing"].attrs["valid_max"] = np.double(360)
        xds["wave_bearing"].attrs["coordinates"] = "time"
        xds["wave_bearing"].attrs["grid_mapping"] = "crs"
        xds["wave_bearing"].attrs["coverage_content_type"] = "physicalMeasurement"

        # Set wind_bearing attributes
        xds["wind_bearing"].attrs["long_name"] = "wind from direction in degrees"
        xds["wind_bearing"].attrs["standard_name"] = "sea_surface_wind_wave_from_direction"
        xds["wind_bearing"].attrs["units"] = "degrees"
        xds["wind_bearing"].attrs["comment"] = "wind from direction in degrees"
        xds["wind_bearing"].attrs["valid_min"] = np.double(0)
        xds["wind_bearing"].attrs["valid_max"] = np.double(360)
        xds["wind_bearing"].attrs["coordinates"] = "time"
        xds["wind_bearing"].attrs["grid_mapping"] = "crs"
        xds["wind_bearing"].attrs["coverage_content_type"] = "physicalMeasurement"

        # Set lon attributes
        xds["lon"].attrs["long_name"] = "Longitude"
        xds["lon"].attrs["standard_name"] = "longitude"
        xds["lon"].attrs["short_name"] = "lon"
        xds["lon"].attrs["units"] = "degrees_east"
        xds["lon"].attrs["axis"] = "X"
        xds["lon"].attrs["valid_min"] = np.double(-180.0)
        xds["lon"].attrs["valid_max"] = np.double(180.0)
        xds["lon"].attrs["grid_mapping"] = "crs"

        # Set lat attributes
        xds["lat"].attrs["long_name"] = "Latitude"
        xds["lat"].attrs["standard_name"] = "latitude"
        xds["lat"].attrs["short_name"] = "lat"
        xds["lat"].attrs["units"] = "degrees_north"
        xds["lat"].attrs["axis"] = "Y"
        xds["lat"].attrs["valid_min"] = np.double(-90.0)
        xds["lat"].attrs["valid_max"] = np.double(90.0)
        xds["lat"].attrs["grid_mapping"] = "crs"

        # # add container variables that contain no data
        # xds = xds.assign(**dict(crs=False, instrument=False))

        # # Set crs attributes
        # xds['crs'].attrs['grid_mapping_name'] = 'latitude_longitude'
        # xds['crs'].attrs['inverse_flattening'] = 298.257223563
        # xds['crs'].attrs['long_name'] = 'Coordinate Reference System'
        # xds['crs'].attrs['semi_major_axis'] = '6378137.0'
        # xds['crs'].attrs['epsg_code'] = 'EPSG:4326'
        # xds['crs'].attrs['comment'] = 'http://www.opengis.net/def/crs/EPSG/0/4326'

        # xds['instrument'].attrs['long_name'] = 'Direction-finding high frequency radar antenna'
        # xds['instrument'].attrs['sensor_type'] = 'Direction-finding high frequency radar antenna'
        # xds['instrument'].attrs['make_model'] = self.metadata['Manufacturer']
        # xds['instrument'].attrs['serial_number'] = 1

        return xds

    def to_netcdf(self, filename, prepend_extension=False, enhance=True):
        """
        Create a compressed netCDF4 (.nc) file from the radial instance

        Args:
            filename (str or path.Path):
                User defined filename of radial file you want to save
            prepend_extension (bool, optional):
                Prepend a descriptive term (ranged or averaged) to the .nc extension. Defaults to False.
            enhance (bool, optional):
                Rename variable names to meaningful names and add attributes. Defaults to True.
        """
        filename = Path(filename)
        os.makedirs(filename.parent.resolve(), exist_ok=True)

        # If the filename does not have a .nc extension, we will add one.
        if ".nc" not in str(filename):
            filename = filename.with_suffix(".nc")

        # If the outputted file exists already, delete the existing file
        if os.path.isfile(filename):
            os.remove(filename)

        # Convert pandas dataframe to xarray dataset using built-in pandas to_xarray function
        xds = self.to_xarray(enhance=enhance)

        # Check if dataset has distance_from_origin in coordinates. We will prepend the .nc extension
        # with the appropriate name depending on whether the wave file is averaged or arranged by distance
        if prepend_extension:
            if "distance_from_origin" in xds.coords:
                pre_ext = "ranged"
            else:
                pre_ext = "averaged"
            # Change the extension to reflect the type of wave file
            filename = filename.with_suffix(f".{pre_ext}.nc")

        # Pass through make_encoding function fo automatically
        encoding = make_encoding(xds, comp_level=4, fillvalue=-999.0)
        encoding["time"] = dict(zlib=False, _FillValue=None)

        # Convert files to netcdf
        xds.to_netcdf(filename, encoding=encoding, format="netCDF4", engine="netcdf4", unlimited_dims=["time"])
