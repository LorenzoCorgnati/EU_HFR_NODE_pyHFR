import logging
import datetime as dt
from dateutil.relativedelta import relativedelta
import math
import numpy as np
import xarray as xr
import netCDF4
import pandas as pd
from pyproj import Geod
from shapely.geometry import Point
import geopandas as gpd
import re
import io
from common import fileParser
from collections import OrderedDict
from calc import gridded_index, true2mathAngle, dms2dd, evaluateGDOP, createLonLatGridFromBB, createLonLatGridFromBBwera, createLonLatGridFromTopLeftPointWera
import json


logger = logging.getLogger(__name__)

# def concatenate_totals(radial_list):
#     """
#     This function takes a list of radial files. Loads them all separately using the Radial object and then combines
#     them along the time dimension using xarrays built-in concatenation routines.
#     :param radial_list: list of radial files that you want to concatenate
#     :return: radial files concatenated into an xarray dataset by range, bearing, and time
#     """
#
#     totals_dict = {}
#     for each in sorted(radial_list):
#         total = Total(each, multi_dimensional=True)
#         totals_dict[total.file_name] = total.ds
#
#     ds = xr.concat(totals_dict.values(), 'time')
#     return ds

def buildEHNtotalFilename(networkID,ts,ext):
    """
    This function builds the filename for total files according to 
    the structure used by the European HFR Node, i.e. networkID-Total_YYYY_MM_DD_hhmm.
    
    INPUT:
        networkID: ID of the HFR network
        ts: timestamp as datetime object
        ext: file extension
        
    OUTPUT:
        totFilename: filename for total file.
    """
    # Get the time related part of the filename
    timeStr = ts.strftime("_%Y_%m_%d_%H%M")
    
    # Build the filename
    totFilename = networkID + '-Total' + timeStr + ext
    
    return totFilename

def buildEHNtotalFolder(basePath,ts,vers):
    """
    This function builds the folder structure for storing total files according to 
    the structure used by the European HFR Node, i.e. YYYY/YYYY_MM/YYYY_MM_DD/.
    
    INPUT:
        basePath: base path
        ts: timestamp as datetime object
        vers: version of the data model
        
    OUTPUT:
        totFolder: folder path for storing total files.
    """
    # Strip trailing slash character
    basePath = basePath.rstrip('/')
    
    # Get the time related part of the path
    timeStr = ts.strftime("/%Y/%Y_%m/%Y_%m_%d/")
    
    # Build the folder path
    totFolder = basePath + '/' + vers + timeStr
    
    return totFolder

def radBinsInSearchRadius(cell,radial,sR,g):
    """
    This function finds out which radial bins are within the spatthresh of the
    origin grid cell.
    The WGS84 CRS is used for distance calculations.
    
    INPUT:
        cell: Series containing longitudes and latitudes of the origin grid cells
        radial: Radial object
        sR: search radius in meters
        g: Geod object according to the Total CRS
        
    OUTPUT:
        radInSr: list of the radial bins falling within the search radius of the
                 origin grid cell.
    """
    # Convert grid cell Series and radial bins DataFrame to numpy arrays
    cell = cell.to_numpy()
    radLon = radial.data['LOND'].to_numpy()
    radLat = radial.data['LATD'].to_numpy() 
    # Evaluate distances between origin grid cells and radial bins
    az12,az21,cellToRadDist = g.inv(len(radLon)*[cell[0]],len(radLat)*[cell[1]],radLon,radLat)
    # Figure out which radial bins are within the spatthresh of the origin grid cell
    radInSR = np.where(cellToRadDist < sR)[0].tolist()
    
    return radInSR


def totalLeastSquare(VelHeadStd):
    """
    This function calculates the u/v components of a total vector from 2 to n 
    radial vector components using weighted Least Square method.
    
    INPUT:
        VelHeadStd: DataFrame containing contributor radial velocities, bearings
                    and standard deviations
        
    OUTPUT:
        u: U component of the total vector
        v: V component of the total vector
        C: covariance matrix
        Cgdop: covariance matrix assuming uniform unit errors for all radials (i.e. all radial std=1)
    """
    # Convert angles from true convention to math convention
    VelHeadStd['HEAD'] = true2mathAngle(VelHeadStd['HEAD'].to_numpy())
    
    # Form the design matrix (i.e. the angle matrix)
    A = np.stack((np.array([np.cos(np.deg2rad(VelHeadStd['HEAD']))/VelHeadStd['STD']]),np.array([np.sin(np.deg2rad(VelHeadStd['HEAD']))/VelHeadStd['STD']])),axis=-1)[0,:,:]
    
    # Form the velocity vector
    b = (VelHeadStd['VELO'].to_numpy())/VelHeadStd['STD']    
    
    # Evaluate the covariance matrix C (variance(U) = C(1,1) and variance(V) = C(2,2))
    C = np.linalg.inv(np.matmul(A.T, A))
    
    # Calculate the u and v for the total vector
    a = np.matmul(C, np.matmul(A.T, b))
    u = a[0]
    v = a[1]    
    
    # Form the design matrix for GDOP evaluation (i.e. setting all radial std to 1)
    Agdop = np.stack((np.array([np.cos(np.deg2rad(VelHeadStd['HEAD']))]),np.array([np.sin(np.deg2rad(VelHeadStd['HEAD']))])),axis=-1)[0,:,:]
    
    # Evaluate the covariance matrix Cgdop for GDOP evaluation (i.e. setting all radial std to 1)
    Cgdop = np.linalg.inv(np.matmul(Agdop.T, Agdop))
    
    return u, v, C, Cgdop


def makeTotalVector(rBins,rDF):
    """
    This function combines radial contributions to get the total vector for each
    grid cell.
    The weighted Least Square method is used for combination.
    
    INPUT:
        rBins: Series containing contributing radial indices.
        rDF: DataFrame containing input Radials.
        
    OUTPUT:
        totalData: Series containing u/v components and related errors of 
                   total vector for each grid cell.
    """
    # set minimum number of contributing radial sites
    minContrSites = 2
    # set minimum number of contributing radial vectors
    minContrRads = 3
    
    # create output total Series
    totalData = pd.Series(np.nan,index=range(6))
    # only consider contributing radial sites
    contrRad = rBins[rBins.str.len() != 0]
    # check if there are at least two contributing radial sites
    if contrRad.size >= minContrSites:
        # loop over contributing radial indices for collecting velocities and angles
        contributions = pd.DataFrame()
        for idx in contrRad.index:
            contrVel = rDF.loc[idx]['Radial'].data.VELO[contrRad[idx]]                                      # pandas Series
            contrHead = rDF.loc[idx]['Radial'].data.HEAD[contrRad[idx]]                                     # pandas Series
            if 'ETMP' in rDF.loc[idx]['Radial'].data.columns:
                contrStd = rDF.loc[idx]['Radial'].data.ETMP[contrRad[idx]]                                  # pandas Series
            elif 'HCSS' in rDF.loc[idx]['Radial'].data.columns:
                contrStd = rDF.loc[idx]['Radial'].data.HCSS[contrRad[idx]]                                  # pandas Series   
            contributions = pd.concat([contributions, pd.concat([contrVel,contrHead,contrStd], axis=1)])    # pandas DataFrame        
                    
        # Rename ETMP column to STD (Codar radial case)
        if 'ETMP' in contributions.columns:
            contributions = contributions.rename(columns={"ETMP": "STD"})
        # Rename HCSS column to STD (WERA radial case) and squareroot the values
        elif 'HCSS' in contributions.columns:
            contributions = contributions.rename(columns={"HCSS": "STD"})
            contributions['STD'] = contributions['STD'].apply(math.sqrt())
            
        # Only keep contributing radials with valid standard deviation values (i.e. different from NaN and 0)
        contributions = contributions[contributions.STD.notnull()]
        contributions = contributions[contributions.STD != 0]
        
        # check if there are at least three contributing radial vectors
        if len(contributions.index) >= minContrRads:
            # combine radial contributions to get total vector for the current grid cell
            u, v, C, Cgdop = totalLeastSquare(contributions)
            
            # populate Total Series
            totalData.loc[0] = u                                            # VELU
            totalData.loc[1] = v                                            # VELV
            totalData.loc[2] = np.sqrt(u**2 + v**2)                         # VELO
            totalData.loc[3] = (360 + np.arctan2(u,v) * 180/np.pi) % 360    # HEAD
            totalData.loc[4] = math.sqrt(C[0,0])                            # UQAL
            totalData.loc[5] = math.sqrt(C[1,1])                            # VQAL
            totalData.loc[6] = C[0,1]                                       # CQAL
            totalData.loc[7] = math.sqrt(Cgdop.trace())                     # GDOP
            totalData.loc[8] = len(contributions.index)                     # NRAD
            
    return totalData


def combineRadials(rDF,gridGS,sRad,gRes,tStp,minContrSites=2):
    """
    This function generataes total vectors from radial measurements using the
    weighted Least Square method for combination.
    
    INPUT:
        rDF: DataFrame containing input Radials; indices must be the site codes.
        gridGS: GeoPandas GeoSeries containing the longitude/latitude pairs of all
            the points in the grid
        sRad: search radius for combination in meters.
        gRes: grid resoultion in meters
        tStp: timestamp in datetime format (YYYY-MM-DD hh:mm:ss)
        minContrSites: minimum number of contributing radial sites (default to 2)
        
    OUTPUT:
        Tcomb: Total object generated by the combination
        warn: string containing warnings related to the success of the combination
    """
    # Initialize empty warning string
    warn = ''

    # Create empty total with grid
    Tcomb = Total(grid=gridGS)

    # Check if there are enough contributing radial sites
    if rDF.size >= minContrSites:
        # Fill site_source DataFrame with contributing radials information
        siteNum = 0    # initialization of site number
        for Rindex, Rrow in rDF.iterrows():
            siteNum = siteNum + 1
            rad = Rrow['Radial']
            thisRadial = pd.DataFrame(index=[Rindex],columns=['#', 'Name', 'Lat', 'Lon', 'Coverage(s)', 'RngStep(km)', 'Pattern', 'AntBearing(NCW)'])
            thisRadial['#'] = siteNum
            thisRadial['Name'] = Rindex
            if rad.is_wera:
                thisRadial['Lon'] = dms2dd(list(map(int,rad.metadata['Longitude(deg-min-sec)OfTheCenterOfTheReceiveArray'].split('-')))) 
                thisRadial['Lat'] = dms2dd(list(map(int,rad.metadata['Latitude(deg-min-sec)OfTheCenterOfTheReceiveArray'][:-2].split('-'))))            
                if rad.metadata['Latitude(deg-min-sec)OfTheCenterOfTheReceiveArray'][-1] == 'S':
                    thisRadial['Lat'] = -thisRadial['Lat']
                thisRadial['Coverage(s)'] = float(rad.metadata['ChirpRate'].replace('S','')) * int(rad.metadata['Samples'])
                thisRadial['RngStep(km)'] = float(rad.metadata['Range'].split()[0])
                thisRadial['Pattern'] = 'Internal'
                thisRadial['AntBearing(NCW)'] = float(rad.metadata['TrueNorth'].split()[0])
            else:        
                thisRadial['Lat'] = float(rad.metadata['Origin'].split()[0])
                thisRadial['Lon'] = float(rad.metadata['Origin'].split()[1])
                thisRadial['Coverage(s)'] = float(rad.metadata['TimeCoverage'].split()[0])
                if 'RangeResolutionKMeters' in rad.metadata:
                    thisRadial['RngStep(km)'] = float(rad.metadata['RangeResolutionKMeters'].split()[0])
                elif 'RangeResolutionMeters' in rad.metadata:
                    thisRadial['RngStep(km)'] = float(rad.metadata['RangeResolutionKMeters'].split()[0]) * 0.001
                thisRadial['Pattern'] = rad.metadata['PatternType'].split()[0]
                thisRadial['AntBearing(NCW)'] = float(rad.metadata['AntennaBearing'].split()[0])
            Tcomb.site_source = pd.concat([Tcomb.site_source, thisRadial])
                
        # Insert timestamp
        Tcomb.time = tStp
    
        # Fill Total with some metadata
        Tcomb.metadata['TimeZone'] = rad.metadata['TimeZone']   # trust all radials have the same, pick from the last radial
        Tcomb.metadata['AveragingRadius'] = str(sRad/1000) + ' km'
        Tcomb.metadata['GridAxisOrientation'] = '0.0 DegNCW'
        Tcomb.metadata['GridSpacing'] = str(gRes/1000) + ' km'
    
        # Create Geod object according to the Total CRS
        g = Geod(ellps=Tcomb.metadata['GreatCircle'].split()[0])
    
        # Create DataFrame for storing indices of radial bins falling within the search radius of each grid cell
        combineRadBins = pd.DataFrame(columns=range(len(Tcomb.data.index)))
    
        # Figure out which radial bins are within the spatthresh of each grid cell
        for Rindex, Rrow in rDF.iterrows():
            rad = Rrow['Radial']         
            thisRadBins = Tcomb.data.loc[:,['LOND','LATD']].apply(lambda x: radBinsInSearchRadius(x,rad,sRad,g),axis=1)
            combineRadBins.loc[Rindex] = thisRadBins
    
        # Loop over grid points and pull out contributing radial vectors
        combineRadBins = combineRadBins.T
        totData = combineRadBins.apply(lambda x: makeTotalVector(x,rDF), axis=1)
    
        # Assign column names to the combination DataFrame
        totData.columns = ['VELU', 'VELV','VELO','HEAD','UQAL','VQAL','CQAL','GDOP','NRAD']
    
        # Fill Total with combination results
        Tcomb.data[['VELU', 'VELV','VELO','HEAD','UQAL','VQAL','CQAL','GDOP','NRAD']] = totData
        
        # Get the indexes of grid cells without total vectors
        indexNoVec = Tcomb.data[Tcomb.data['VELU'].isna()].index
        # Delete these row indexes from DataFrame
        Tcomb.data.drop(indexNoVec , inplace=True)
        Tcomb.data.reset_index(level=None, drop=False, inplace=True)    # Set drop=True if the former indices are not necessary
        
    else:
        warn = 'No combination performed: not enough contributing radial sites'
    
    return Tcomb, warn


class Total(fileParser):
    """
    Totals Subclass.

    This class should be used when loading CODAR (.tuv) and WERA (.cur_asc) total files.
    This class utilizes the generic LLUV and CUR classes.
    """
    def __init__(self, fname='', replace_invalid=True, grid=gpd.GeoSeries(), empty_total=False):

        if not fname:
            empty_total = True
            replace_invalid = False
            
        super().__init__(fname)
        for key in self._tables.keys():
            table = self._tables[key]
            if 'LLUV' in table['TableType']:
                self.data = table['data']                
            elif 'src' in table['TableType']:
                self.diagnostics_source = table['data']
            elif 'CUR' in table['TableType']:
                self.cur_data = table['data']
                
        if 'SiteSource' in self.metadata.keys():
            if not self.is_wera:
                table_data = u''
                for ss in self.metadata['SiteSource']:
                    if '%%' in ss:
                        rep = {' comp': '_comp', ' Distance': '_Distance',' Ratio': '_Ratio',' (dB)': '_(dB)',' Width': '_Width', ' Resp': '_Resp', 'Value ': 'Value_','FOL ':'FOL_' }
                        rep = dict((re.escape(k), v) for k, v in rep.items())
                        pattern = re.compile('|'.join(rep.keys()))
                        ss_header = pattern.sub(lambda m: rep[re.escape(m.group(0))], ss).strip('%% SiteSource \n')
                    else:
                        ss = ss.replace('%SiteSource:', '').strip()
                        ss = ss.replace('Radial', '').strip()
                        table_data += '{}\n'.format(ss)
                # use pandas read_csv because it interprets the datatype for each column of the csv
                tdf = pd.read_csv(
                    io.StringIO(table_data),
                    sep=' ',
                    header=None,
                    names=ss_header.split(),
                    skipinitialspace=True
                )
                self.site_source = tdf
                
        # Evaluate GDOP for total files
        if hasattr(self, 'site_source'):
            # Get the site coordinates
            siteLon = self.site_source['Lon'].values.tolist()
            siteLat = self.site_source['Lat'].values.tolist()  
            # Create Geod object according to the Total CRS, if defined. Otherwise use WGS84 ellipsoid
            if self.metadata['GreatCircle']:
                g = Geod(ellps=self.metadata['GreatCircle'].split()[0].replace('"',''))                  
            else:
                g = Geod(ellps='WGS84')
                self.metadata['GreatCircle'] = '"WGS84"' + ' ' + str(g.a) + '  ' + str(1/g.f)
            self.data['GDOP'] = self.data.loc[:,['LOND','LATD']].apply(lambda x: evaluateGDOP(x,siteLon, siteLat, g),axis=1)                
        elif hasattr(self, 'data'):
            self.data['GDOP'] = np.nan
            
        # Evaluate the number of contributing radials (NRAD) for CODAR total files
        if hasattr(self, 'is_wera'):
            if not self.is_wera:
                if hasattr(self, 'data'):
                    self.data['NRAD'] = self.data.loc[:, self.data.columns.str.contains('S.*CN')].sum(axis=1)
            

        if replace_invalid:
            self.replace_invalid_values()
            
        if empty_total:
            self.empty_total()
            
        # if mask_over_land:
        #     self.mask_over_land()

        if not grid.empty:
            self.initialize_grid(grid)
            
    
    def empty_total(self):
        """
        Create an empty Total object. The empty Total object can be created by setting 
        the geographical grid.
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
            elif 'rads' in table['TableType']:
                self.diagnostics_radial.drop(self.diagnostics_radial.index[:], inplace=True)
                self._tables[key]['data'] = self.diagnostics_radial
            elif 'rcvr' in table['TableType']:
                self.diagnostics_hardware.drop(self.diagnostics_hardware.index[:], inplace=True)
                self._tables[key]['data'] = self.diagnostics_hardware
            elif 'RINF' in table['TableType']:
                self.range_information.drop(self.range_information.index[:], inplace=True)
                self._tables[key]['data'] = self.range_information
            elif 'CUR' in table['TableType']:
                self.cur_data.drop(self.cur_data.index[:], inplace=True)
                self._tables[key]['data'] = self.cur_data
                
        if not hasattr(self, 'data'):
            self.data = pd.DataFrame()
        
        if hasattr(self, 'site_source'):
            self.site_source.drop(self.site_source.index[:], inplace=True)
        else:
            self.site_source = pd.DataFrame()
            
    
    def initialize_grid(self, gridGS):
        """
        Initialize the geogprahic grid for filling the LOND and LATD columns of the 
        Total object data DataFrame.
        
        INPUT:
            gridGS: GeoPandas GeoSeries containing the longitude/latitude pairs of all
                the points in the grid
                
        OUTPUT:
            DataFrame with filled LOND and LATD columns.
        """
        
        # initialize data DataFrame with column names
        self.data = pd.DataFrame(columns=['LOND', 'LATD', 'VELU', 'VELV', 'VELO', 'HEAD', 'UQAL', 'VQAL', 'CQAL', 'GDOP', 'NRAD'])
        
        # extract longitudes and latitude from grid GeoSeries and insert them into data DataFrame
        self.data['LOND'] = gridGS.x
        self.data['LATD'] = gridGS.y
        
        # add metadata about datum and CRS
        self.metadata = OrderedDict()
        self.metadata['GreatCircle'] = ''.join(gridGS.crs.ellipsoid.name.split()) + ' ' + str(gridGS.crs.ellipsoid.semi_major_metre) + '  ' + str(gridGS.crs.ellipsoid.inverse_flattening)

    
    def mask_over_land(self, subset=True):
        """
        This function masks the total vectors lying on land.        
        Total vector coordinates are checked against a reference file containing information 
        about which locations are over land or in an unmeasurable area (for example, behind an 
        island or point of land). 
        The GeoPandas "naturalearth_lowres" is used as reference.        
        The native CRS of the Total is used for distance calculations.
        If "subset"  option is set to True, the total vectors lying on land are removed.
        
        INPUT:
            subset: option enabling the removal of total vectors on land (if set to True)
            
        OUTPUT:
            waterIndex: list containing the indices of total vectors lying on water.
        """
        # Load the reference file (GeoPandas "naturalearth_lowres")
        land = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        # land = land[land['continent'] == 'North America']

        # Build the GeoDataFrame containing total points
        geodata = gpd.GeoDataFrame(
            self.data[['LOND', 'LATD']],
            crs=land.crs.srs.upper(),
            geometry=[
                Point(xy) for xy in zip(self.data.LOND.values, self.data.LATD.values)
            ]
        )
        # Join the GeoDataFrame containing total points with GeoDataFrame containing leasing areas
        geodata = gpd.tools.sjoin(geodata, land, how='left', predicate='intersects')

        # All data in the continent column that lies over water should be nan.
        waterIndex = geodata['continent'].isna()

        if subset:
            # Subset the data to water only
            self.data = self.data.loc[waterIndex].reset_index()
        else:
            return waterIndex
        
        
    def to_xarray_multidimensional(self, lon_min=None, lon_max=None, lat_min=None, lat_max=None, grid_res=None):
        """
        This function creates a dictionary of xarray DataArrays containing the variables
        of the Total object multidimensionally expanded along the coordinate axes (T,Z,Y,X).  
        The coordinate axes are set as (TIME, DEPTH, LATITUDE, LONGITUDE).
        The coordinate limits and steps are taken from total metadata when possible.
        Only longitude and latitude limits and grid resolution can be specified by the user.
        Some refinements are performed on Codar data in order to comply CF convention
        for positive velocities and to have velocities in m/s.
        The generated dictionary is attached to the Total object, named as xdr.
        
        INPUT:
            lon_min: minimum longitude value in decimal degrees (if None it is taken from Total metadata)
            lon_miax: maximum longitude value in decimal degrees (if None it is taken from Total metadata)
            lat_min: minimum latitude value in decimal degrees (if None it is taken from Total metadata)
            lat_miax: maximum latitude value in decimal degrees (if None it is taken from Total metadata)
            grid_res: grid resolution in meters (if None it is taken from Total metadata)
            
        OUTPUT:

        """
        # Initialize empty dictionary
        xdr = OrderedDict()
        
        # process Codar radial
        if not self.is_wera:
            # Check longitude limits   
            if lon_min is None:
                if 'BBminLongitude' in self.metadata:
                    lon_min = float(self.metadata['BBminLongitude'].split()[0])
                else:
                    lon_min = self.data.LOND.min()
            if lon_max is None:
                if 'BBmaxLongitude' in self.metadata:
                    lon_max = float(self.metadata['BBmaxLongitude'].split()[0])
                else:
                    lon_max = self.data.LOND.max()
            # Check latitude limits   
            if lat_min is None:
                if 'BBminLatitude' in self.metadata:
                    lat_min = float(self.metadata['BBminLatitude'].split()[0])
                else:
                    lat_min = self.data.LATD.min()
            if lat_max is None:
                if 'BBmaxLatitude' in self.metadata:
                    lat_max = float(self.metadata['BBmaxLatitude'].split()[0])
                else:
                    lat_max = self.data.LATD.max()                
            # Get grid resolution
            if grid_res is None:                
                if 'GridSpacing' in self.metadata:
                    grid_res = float(self.metadata['GridSpacing'].split()[0]) * 1000
                else:
                    grid_res = float(1)            
            
            # Generate grid coordinates
            gridGS = createLonLatGridFromBB(lon_min, lon_max, lat_min, lat_max, grid_res)
                    
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
        lat_dim = np.flipud(np.unique(gridGS.y.to_numpy()))
        # manage antimeridian crossing
        lon_dim = np.concatenate((lon_dim[lon_dim>=0],lon_dim[lon_dim<0]))
        
        # Get the longitude and latitude values of the total measurements
        unqLon = np.sort(np.unique(self.data['LOND']))
        unqLat = np.flipud(np.sort(np.unique(self.data['LATD'])))
        
        # Insert unqLon and unqLat values to replace the closest in lon_dim and lat_dim 
        replaceIndLon = abs(unqLon[None, :] - lon_dim[:, None]).argmin(axis=0).tolist()
        replaceIndLat = abs(unqLat[None, :] - lat_dim[:, None]).argmin(axis=0).tolist()
        lon_dim[replaceIndLon] = unqLon
        lat_dim[replaceIndLat] = unqLat            

        # Create total grid from longitude and latitude
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

        # Drop LOND and LATD variables (they are set as coordinates of the DataSet)
        d.pop('LOND')
        d.pop('LATD')

        # Refine Codar and combined data
        if not self.is_wera:
            # Scale velocities to be in m/s (only for Codar or combined totals)
            toMs = ['VELU', 'VELV', 'VELO', 'UQAL', 'VQAL','CQAL']
            for t in toMs:
                if t in d:
                    d[t] = d[t]*0.01

        # Evaluate timestamp as number of days since 1950-01-01T00:00:00Z
        timeDelta = self.time - dt.datetime.strptime('1950-01-01T00:00:00Z','%Y-%m-%dT%H:%M:%SZ')
        ncTime = timeDelta.days + timeDelta.seconds / (60*60*24)
        
        # Add all variables as xarray
        for k, v in d.items():
            xdr[k] = xr.DataArray(v,
                                  dims={'TIME': v.shape[0], 'DEPTH': v.shape[1], 'LATITUDE': v.shape[2], 'LONGITUDE': v.shape[3]},
                                  coords={'TIME': [ncTime],
                                          'DEPTH': [0],
                                          'LATITUDE': lat_dim,
                                          'LONGITUDE': lon_dim})
        
        # Add DataArray for coordinate variables
        xdr['TIME'] = xr.DataArray(ncTime,
                                 dims={'TIME': len(pd.date_range(self.time, periods=1))},
                                 coords={'TIME': [ncTime]})
        xdr['DEPTH'] = xr.DataArray(0,
                                 dims={'DEPTH': 1},
                                 coords={'DEPTH': [0]})
        xdr['LATITUDE'] = xr.DataArray(lat_dim,
                                       dims={'LATITUDE': lat_dim},
                                       coords={'LATITUDE': lat_dim})
        xdr['LONGITUDE'] = xr.DataArray(lon_dim,
                                        dims={'LONGITUDE': lon_dim},
                                        coords={'LONGITUDE': lon_dim})  
        
        # Attach the dictionary to the Total object
        self.xdr = xdr
        
        return
    
    
    def apply_ehn_datamodel(self, network_data, station_data, version):
        """
        This function applies the European standard data model developed in the
        framework of the EuroGOOS HFR Task Team to the Total object.
        The Total object content is stored into an xarray Dataset built from the
        xarray DataArrays created by the Total method to_xarray_multidimensional.
        Variable data types and data packing information are collected from
        "Data_Models/EHN/Totals/Total_Data_Packing.json" file.
        Variable attribute schema is collected from 
        "Data_Models/EHN/Totals/Total_Variables.json" file.
        Global attribute schema is collected from 
        "Data_Models/EHN/Global_Attributes.json" file.
        Global attributes are created starting from Total object metadata and from 
        DataFrames containing the information about HFR network and radial stations
        read from the EU HFR NODE database.
        The generated xarray Dataset is attached to the Total object, named as xds.
        
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
        maxsiteSize = 50
        refmaxSize = 10
        maxinstSize = 10
        
        # Get data packing information per variable
        f = open('Data_Models/EHN/Totals/Total_Data_Packing.json')
        dataPacking = json.loads(f.read())
        f.close()
        
        # Get variable attributes
        f = open('Data_Models/EHN/Totals/Total_Variables.json')
        totVariables = json.loads(f.read())
        f.close()
        
        # Get global attributes
        f = open('Data_Models/EHN/Global_Attributes.json')
        globalAttributes = json.loads(f.read())
        f.close()
        
        # Drop unnecessary DataArrays from the DataSet
        toDrop = ['VFLG', 'XDST', 'YDST', 'RNGE', 'BEAR','S*CN', 'NRAD', 'VELO', 'HEAD','index']
        for t in toDrop:
            if t in self.xdr:
                self.xdr.pop(t)
            
        # Add coordinate reference system to the dictionary
        self.xdr['crs'] = xr.DataArray(np.int(0), )
        
        # Rename velocity related and quality related variables
        self.xdr['EWCT'] = self.xdr.pop('VELU')
        self.xdr['NSCT'] = self.xdr.pop('VELV')
        self.xdr['EWCS'] = self.xdr.pop('UQAL')
        self.xdr['NSCS'] = self.xdr.pop('VQAL')
        self.xdr['CCOV'] = self.xdr.pop('CQAL')
        
        # Add antenna related variables to the dictionary
        # Number of antennas        
        nRX = np.asfarray(station_data['number_of_receive_antennas'].to_numpy())
        nRX = np.pad(nRX, (0, maxsiteSize - len(nRX)), 'constant',constant_values=(np.nan,np.nan))
        nTX = np.asfarray(station_data['number_of_transmit_antennas'].to_numpy())
        nTX = np.pad(nTX, (0, maxsiteSize - len(nTX)), 'constant',constant_values=(np.nan,np.nan))
        self.xdr['NARX'] = xr.DataArray([nRX], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xdr['NATX'] = xr.DataArray([nTX], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        
        # Longitude and latitude of antennas
        siteLat = station_data['site_lat'].to_numpy()
        siteLat = np.pad(siteLat, (0, maxsiteSize - len(siteLat)), 'constant',constant_values=(np.nan,np.nan))
        siteLon = station_data['site_lon'].to_numpy()
        siteLon = np.pad(siteLon, (0, maxsiteSize - len(siteLon)), 'constant',constant_values=(np.nan,np.nan))
        self.xdr['SLTR'] = xr.DataArray([siteLat], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xdr['SLNR'] = xr.DataArray([siteLon], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xdr['SLTT'] = xr.DataArray([siteLat], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xdr['SLNT'] = xr.DataArray([siteLon], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        
        # Codes of antennas
        antCode = np.array([site.encode() for site in station_data['station_id'].tolist()])
        antCode = np.pad(antCode, (0, maxsiteSize - len(station_data)), 'constant',constant_values=('',''))
        self.xdr['SCDR'] = xr.DataArray(np.array([antCode]), dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xdr['SCDR'].encoding['char_dim_name'] = 'STRING' + str(len(station_data['station_id'].to_numpy()[0]))
        self.xdr['SCDT'] = xr.DataArray(np.array([antCode]), dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXSITE': maxsiteSize})
        self.xdr['SCDT'].encoding['char_dim_name'] = 'STRING' + str(len(station_data['station_id'].to_numpy()[0]))
                
        # Add SDN namespace variables to the dictionary
        siteCode = ('%s' % network_data.iloc[0]['network_id']).encode()
        self.xdr['SDN_CRUISE'] = xr.DataArray([siteCode], dims={'TIME': len(pd.date_range(self.time, periods=1))})
        self.xdr['SDN_CRUISE'].encoding['char_dim_name'] = 'STRING' + str(len(siteCode))
        platformCode = ('%s' % network_data.iloc[0]['network_id'] + '-Total').encode()
        self.xdr['SDN_STATION'] = xr.DataArray([platformCode], dims={'TIME': len(pd.date_range(self.time, periods=1))})
        self.xdr['SDN_STATION'].encoding['char_dim_name'] = 'STRING' + str(len(platformCode))
        ID = ('%s' % platformCode.decode() + '_' + self.time.strftime('%Y-%m-%dT%H:%M:%SZ')).encode()
        self.xdr['SDN_LOCAL_CDI_ID'] = xr.DataArray([ID], dims={'TIME': len(pd.date_range(self.time, periods=1))})
        self.xdr['SDN_LOCAL_CDI_ID'].encoding['char_dim_name'] = 'STRING' + str(len(ID))
        sdnEDMO = np.asfarray(pd.concat([network_data['EDMO_code'],station_data['EDMO_code']]).unique())
        sdnEDMO = np.pad(sdnEDMO, (0, maxinstSize - len(sdnEDMO)), 'constant',constant_values=(np.nan,np.nan))
        self.xdr['SDN_EDMO_CODE'] = xr.DataArray([sdnEDMO], dims={'TIME': len(pd.date_range(self.time, periods=1)), 'MAXINST': maxinstSize})
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
            self.xds[vv].attrs = totVariables[vv]
            
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
            self.xds[cc].attrs = totVariables[cc]
            
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
        globalAttributes['doa_estimation_method'] = ', '.join(station_data[["station_id", "DoA_estimation_method"]].apply(": ".join, axis=1))
        globalAttributes['calibration_type'] = ', '.join(station_data[["station_id", "calibration_type"]].apply(": ".join, axis=1))
        globalAttributes['last_calibration_date'] = ', '.join(pd.concat([station_data['station_id'],station_data['last_calibration_date'].apply(lambda x: x.strftime('%Y-%m-%dT%H:%M:%SZ'))],axis=1)[["station_id", "last_calibration_date"]].apply(": ".join, axis=1))
        globalAttributes['calibration_link'] = ', '.join(station_data[["station_id", "calibration_link"]].apply(": ".join, axis=1))
        globalAttributes['title'] = network_data.iloc[0]['title']
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
        
        creationDate = dt.datetime.utcnow()
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
            if 'valid_min' in totVariables[vv]:
                if ('scale_factor' in dataPacking[vv]) and ('add_offset' in dataPacking[vv]):
                    self.xds[vv].attrs['valid_min'] = np.float_(((totVariables[vv]['valid_min'] - dataPacking[vv]['add_offset']) / dataPacking[vv]['scale_factor'])).astype(dataPacking[vv]['dtype'])
                else:
                    self.xds[vv].attrs['valid_min'] = np.float_(totVariables[vv]['valid_min']).astype(dataPacking[vv]['dtype'])
            if 'valid_max' in totVariables[vv]:             
                if ('scale_factor' in dataPacking[vv]) and ('add_offset' in dataPacking[vv]):
                    self.xds[vv].attrs['valid_max'] = np.float_(((totVariables[vv]['valid_max'] - dataPacking[vv]['add_offset']) / dataPacking[vv]['scale_factor'])).astype(dataPacking[vv]['dtype'])
                else:
                    self.xds[vv].attrs['valid_max'] = np.float_(totVariables[vv]['valid_max']).astype(dataPacking[vv]['dtype'])
            
        # Encode data types and avoid data packing, valid_min, valid_max and _FillValue for the coordinate variables of the DataSet
        for cc in self.xds.coords:
            if cc in dataPacking:
                if 'dtype' in dataPacking[cc]:
                    self.xds[cc].encoding['dtype'] = dataPacking[cc]['dtype']
                if 'valid_min' in totVariables[cc]:
                    del self.xds[cc].attrs['valid_min']
                if 'valid_max' in totVariables[cc]:
                    del self.xds[cc].attrs['valid_max']
                self.xds[cc].encoding['_FillValue'] = None
               
        return

    
    # def to_multi_dimensional(self, grid_file):
    #     try:
    #         # load grid file
    #         grid = pd.read_csv(grid_file, sep=',', header=None, names=['lon', 'lat'], delim_whitespace=True)
    #         logging.debug('{} - Grid file successfully loaded '.format(grid_file))
    #     except Exception as err:
    #         logging.error('{} - {}. Grid file could not be loaded.'.format(grid_file, err))
    #         return

    #     lon = np.unique(grid['lon'].values.astype(np.float32))
    #     lat = np.unique(grid['lat'].values.astype(np.float32))
    #     [x, y] = np.meshgrid(lon, lat)

    #     logging.debug('{} - Gridding data to 2d grid'.format(grid_file))

    #     # convert 1d data into 2d gridded form. data_dict must be a dictionary.
    #     x_ind, y_ind = gridded_index(x, y, self.data.lon, self.data.lat)

    #     coords = ('time', 'range', 'bearing')

    #     # Intitialize empty xarray dataset
    #     ds = xr.Dataset()
    
    
    def initialize_qc(self):
        """
        Initialize dictionary entry for QC metadata.
        """
        # Initialize dictionary entry for QC metadta
        self.metadata['QCTest'] = {}
        
        
    def qc_ehn_maximum_velocity(self, totMaxSpeed=1.2):
        """
        This test labels total velocity vectors whose module is smaller than a maximum velocity threshold 
        with a “good data” flag. Otherwise the vectors are labeled with a “bad data” flag.
        The ARGO QC flagging scale is used.
        
        This test was defined in the framework of the EuroGOOS HFR Task Team based on the
        Max Speed Threshold test (QC303) from the Integrated Ocean Observing System (IOOS) Quality Assurance of 
        Real-Time Oceanographic Data (QARTOD).
        
        INPUTS:
            totMaxSpeed: maximum velocity in m/s for normal operations                     
        """
        # Set the test name
        testName = 'CSPD_QC'
        
        # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
        self.data.loc[:,testName] = 1
    
        # set bad flag for velocities not passing the test
        if self.is_wera:
            self.data.loc[(self.data['VELO'].abs() > totMaxSpeed), testName] = 4          # velocity in m/s (CRAD)
        else:
            self.data.loc[(self.data['VELO'].abs() > totMaxSpeed*100), testName] = 4      # velocity in cm/s (LLUV)
    
        self.metadata['QCTest'][testName] = 'Velocity Threshold QC Test - Test applies to each vector. ' \
            + 'Threshold=[' + f'maximum velocity={totMaxSpeed} (m/s)]'
            
    def qc_ehn_maximum_variance(self, totMaxVar=1):
        """
        This test labels total velocity vectors whose temporal variances for both U and V
        components are smaller than a maximum variance threshold with a “good data” flag. 
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
            totMaxVar: maximum variance in m2/s2 for normal operations                     
        """
        # Set the test name
        testName = 'VART_QC'
    
        # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
        self.data.loc[:,testName] = 1
    
        # Set bad flag for variances not passing the test
        if self.is_wera:
            self.data.loc[(self.data['UACC']**2 > totMaxVar), testName] = 4             # UACC is the temporal standard deviation of U component in m/s for WERA data
            self.data.loc[(self.data['VACC']**2 > totMaxVar), testName] = 4             # VACC is the temporal standard deviation of V component in m/s for WERA data
        else:
            self.data.loc[((self.data['UQAL']/100)**2 > totMaxVar), testName] = 4       # UQAL is the temporal standard deviation of U component in cm/s for CODAR data
            self.data.loc[((self.data['VQAL']/100)**2 > totMaxVar), testName] = 4       # VQAL is the temporal standard deviation of V component in cm/s for CODAR data
    
        self.metadata['QCTest'][testName] = 'Variance Threshold QC Test - Test applies to each vector. ' \
            + 'Threshold=[' + f'maximum variance={totMaxVar} (m2/s2)]'
        
    def qc_ehn_gdop_threshold(self, maxGDOP=2):
        """
        This test labels total velocity vectors whose GDOP is smaller than a maximum GDOP threshold 
        with a “good data” flag. Otherwise the vectors are labeled with a “bad data” flag.
        The ARGO QC flagging scale is used.
        
        This test was defined in the framework of the EuroGOOS HFR Task Team based on the
        GDOP Threshold test (QC302) from the Integrated Ocean Observing System (IOOS) Quality Assurance of 
        Real-Time Oceanographic Data (QARTOD).
        
        INPUTS:
            maxGDOP: maximum allowed GDOP for normal operations                     
        """
        # Set the test name
        testName = 'GDOP_QC'
        
        # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
        self.data.loc[:,testName] = 1
    
        # set bad flag for velocities not passing the test
        self.data.loc[(self.data['GDOP'] > maxGDOP), testName] = 4
    
        self.metadata['QCTest'][testName] = 'GDOP Threshold QC Test - Test applies to each vector. ' \
            + 'Threshold=[' + f'GDOP threshold={maxGDOP}]'
        
    def qc_ehn_data_density_threshold(self, minContrRad=2):
        """
        This test labels total velocity vectors with a number of contributing radial velocities smaller 
        than the minimum number defined for normal operations with a “good data” flag. 
        Otherwise the vectors are labeled with a “bad data” flag.
        The ARGO QC flagging scale is used.
        
        This test was defined in the framework of the EuroGOOS HFR Task Team based on the
        Data Density Threshold test (QC301) from the Integrated Ocean Observing System (IOOS) Quality Assurance of 
        Real-Time Oceanographic Data (QARTOD).
        
        INPUTS:
            minContrRad: minimum number of contributing radial velocities for normal operations                     
        """
        # Set the test name
        testName = 'DDNS_QC'
        
        # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
        self.data.loc[:,testName] = 1
    
        # set bad flag for velocities not passing the test
        if not self.is_wera:
            if 'NRAD' in self.data.columns:
                self.data.loc[(self.data['NRAD'] < minContrRad), testName] =4
            else:
                self.data.loc[:,testName] = 0
    
        self.metadata['QCTest'][testName] = 'Data Density Threshold QC Test - Test applies to each vector. ' \
            + 'Threshold=[' + f'minimum number of contributing radial velocities={minContrRad}]'
        
    def qc_ehn_temporal_derivative(self, t0, tempDerThr=1):
        """
        This test compares the velocity of each total vector with the velocity of the total vector 
        measured in the previous timestamp at the same location.
        Each vector for which the velocity difference is smaller than the specified threshold for normal 
        operations (tempDerThr), is labeled with a "good data" flag.
        Otherwise the vector is labeled with a “bad data” flag.
        The ARGO QC flagging scale is used.
        
        This test was defined in the framework of the EuroGOOS HFR Task Team based on the 
        Temporal Gradient test (QC206) from the Integrated Ocean Observing System (IOOS) Quality 
        Assurance of Real-Time Oceanographic Data (QARTOD).
        
        INPUTS:
            t0: Total object of the previous timestamp
            tempDerThr: velocity difference threshold in m/s for normal operations
        """
        # Set the test name
        testName = 'VART_QC'
        
        # Check if the previous timestamp total file exists
        if not t0 is None:
            # Merge the data DataFrame of the two Totals and evaluate velocity differences at each location
            mergedDF = self.data.merge(t0.data, on=['LOND', 'LATD'], how='left', suffixes=(None, '_x'), indicator='Exist')
            velDiff = (mergedDF['VELO'] - mergedDF['VELO_x']).abs()

            # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
            self.data.loc[:,testName] = 1

            # Set rows of the DataFrame for QC data as not evaluated (flag = 0) for locations existing in the current total but not in the previous one
            self.data.loc[mergedDF['Exist'] == 'left_only', testName] = 0

            # Set bad flag for vectors not passing the test
            if self.is_wera:
                self.data.loc[(velDiff > tempDerThr), testName] = 4             # velocity in m/s (CUR)
            else:
                self.data.loc[(velDiff > tempDerThr*100), testName] = 4         # velocity in cm/s (LLUV)

        else:
            # Add new column to the DataFrame for QC data by setting every row as not evaluated (flag = 0)
            self.data.loc[:,testName] = 0
        
        self.metadata['QCTest'][testName] = 'Temporal Derivative QC Test - Test applies to each vector. ' \
            + 'Threshold=[' + f'velocity difference threshold={str(tempDerThr)} (m/s)]'
        
    def qc_ehn_overall_qc_flag(self):
        """
        
        This QC test labels total velocity vectors with a ‘good_data” flag if all QC tests are passed.
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


    def file_type(self):
        """
        Return a string representing the type of file this is.
        """
        return 'totals'