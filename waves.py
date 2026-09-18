from calendar import EPOCH
import datetime as dt
import pandas as pd
import re
from pyproj import Geod
import xarray as xr
from common import fileParser
# from nc import make_encoding
import numpy as np
import os
from pathlib import Path
from collections import OrderedDict
import geopandas as gpd

import logging

logger = logging.getLogger(__name__)


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
            avgDistance (float, optional): average distance for Codar averaged files. Defaults to 15 km.
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
                    return
            else:                       # WERA data
                # Create Geod object according to the Total CRS, if defined. Otherwise use WGS84 ellipsoid
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
            The LATITUDE and LONGITUDE values are taken from the Wave object metadata.
            The generated dictionary is attached to the Total object, named as xts.
    
            """
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
                time = df['time'].values                      # TIME axis (length N)
                depth = np.array([0], dtype=float)              # DEPTH axis (length 1)
                coords = {"TIME": ("TIME", time), "DEPTH": ("DEPTH", depth)}

                # Set the columns that become (TIME, DEPTH) variables
                variable_cols = [c for c in df.columns if c not in ('TIME', 'time', 'LOND', 'LATD')]

            for col in variable_cols:
                data = df[col].values[:, np.newaxis]            # (N,) -> (N, 1)
                xts[col] = xr.DataArray(
                    data=data,
                    dims=("TIME", "DEPTH"),
                    coords=coords,
                    name=col,
                )

            # Add DataArray for coordinate variables
            xts['TIME'] = xr.DataArray(time,
                                     dims={'TIME': len(time)},
                                     coords={'TIME': len(time)})
            xts['DEPTH'] = xr.DataArray(0,
                                     dims={'DEPTH': 1},
                                     coords={'DEPTH': [0]})
            xts['LATITUDE'] = xr.DataArray(df['LATD'].iloc[0],
                                           dims={'LATITUDE': df['LATD'].iloc[0]},
                                           coords={'LATITUDE': df['LATD'].iloc[0]})
            xts['LONGITUDE'] = xr.DataArray(df['LOND'].iloc[0],
                                            dims={'LONGITUDE': df['LOND'].iloc[0]},
                                            coords={'LONGITUDE': df['LOND'].iloc[0]})  
            
            # Attach the dictionary to the Total object
            self.xts = xts
            
            return

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
