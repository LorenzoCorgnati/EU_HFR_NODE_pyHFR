import logging
import math
import numpy as np
import xarray as xr
import pandas as pd
from pyproj import Geod
from shapely.geometry import Point
import geopandas as gpd
import re
import io
from common import fileParser
from collections import OrderedDict
from calc import gridded_index, true2mathAngle

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
    
    # ONLY FOR DEBUG PURPOSE
    if pd.isna(u):
        print('SOMETHING WENT WRONG')
    
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
            contrStd = rDF.loc[idx]['Radial'].data.ETMP[contrRad[idx]]                                      # pandas Series
            contributions = pd.concat([contributions, pd.concat([contrVel,contrHead,contrStd], axis=1)])    # pandas DataFrame
                    
        # Rename ETMP column to STD (Codar radial case)
        if 'ETMP' in contributions.columns:
            contributions = contributions.rename(columns={"ETMP": "STD"})
        # Rename HCSS column to STD (WERA radial case) and squareroot the values
        elif 'HCSS' in contributions.columns:
            contributions = contributions.rename(columns={"HCSS": "STD"})
            contributions['STD'] = contributions['STD'].apply(math.sqrt())
        
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
        self.metadata['QCTest'] = []
        
        
    def qc_ehn_maximum_velocity(self, totMaxSpeed=1.2):
        """
        This test labels total velocity vectors whose module is smaller than a maximum velocity threshold 
        with a ???good data??? flag. Otherwise the vector is labeled with a ???bad data??? flag.
        The ARGO QC flagging scale is used.
        
        This test was defined in the framework of the EuroGOOS HFR Task Team based on the
        Max Speed Threshold test (QC303) from the Integrated Ocean Observing System (IOOS) Quality Assurance of 
        Real-Time Oceanographic Data (QARTOD).
        
        INPUTS:
            totMaxSpeed: maximum velocity in m/s for normal operations                     
        """
        # Set the test name
        testName = 'CSPD_QC'
        
        # Evaluate velocity module from U and V components and add it to the data DataFrame
        self.data['VELO'] = np.sqrt(self.data['VELU']**2 + self.data['VELV']**2)
    
        # Add new column to the DataFrame for QC data by setting every row as passing the test (flag = 1)
        self.data.loc[:,testName] = 1
    
        # set bad flag for velocities not passing the test
        if self.is_wera:
            self.data.loc[(self.data['VELO'].abs() > totMaxSpeed), testName] = 4          # velocity in m/s (CRAD)
        else:
            self.data.loc[(self.data['VELO'].abs() > totMaxSpeed*100), testName] = 4      # velocity in cm/s (LLUV)
    
        self.metadata['QCTest'].append((
            f'Velocity Threshold QC Test - Test applies to each vector. Threshold='
            '[ '
            f'maximum velocity={totMaxSpeed} (m/s)]'
        ))


    def file_type(self):
        """
        Return a string representing the type of file this is.
        """
        return 'totals'