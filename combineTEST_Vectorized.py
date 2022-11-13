#!/usr/bin/python3

import datetime as dt
import pandas as pd
import numpy as np
import glob
import io
import logging
import os
from radials import Radial
from totals import Total
from calc import true2mathAngle, dms2dd, createLonLatGridFromBB, createLonLatGridFromBBwera, createLonLatGridFromTopLeftPointWera
from pyproj import Geod
import latlon
import time
import math

def radBinsInSearchRadius(cell,radial,sR,g):
    """
    This function finds out which radial bins are within the spatthresh of each
    grid cell.
    The WGS84 CRS is used for distance calculations.
    
    INPUTS:
        cell: Series containing longitudes and latitudes of the grid cells
        radial: Radial object
        sR: search radius in meters
        g: Geod object according to the Total CRS
        
    OUTPUTS:
        radInSr: Series containing a list of the radial bins falling within the
                 search radius of each grid cell.
    """
    # Convert grid cell Series and radial bins DataFrame to numpy arrays
    cell = cell.to_numpy()
    radLon = radial.data['LOND'].to_numpy()
    radLat = radial.data['LATD'].to_numpy() 
    # Evaluate distances between grid cells and radial bins
    az12,az21,cellToRadDist = g.inv(len(radLon)*[cell[0]],len(radLat)*[cell[1]],radLon,radLat)
    # Figure out which radial bins are within the spatthresh of each grid cell
    radInSR = np.where(cellToRadDist < sR)[0].tolist()
    
    return radInSR


def totalLeastSquare(VelHeadStd):
    """
    This function calculates the u/v components of a total vector from 2 to n 
    radial vector components using weighted Least Square method.
    
    INPUTS:
        VelHeadStd: DataFrame containing contributor radial velocities, bearings
                    and standard deviations
        
    OUTPUTS:
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
    
    INPUTS:
        rBins: Series containing contributing radial indices.
        rDF: DataFrame containing input Radials.
        
    OUTPUTS:
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
            contrVel = rDF.loc[idx]['Radial'].data.VELO[contrRad[idx]]                                  # pandas Series
            contrHead = rDF.loc[idx]['Radial'].data.HEAD[contrRad[idx]]                                 # pandas Series
            contrStd = rDF.loc[idx]['Radial'].data.ETMP[contrRad[idx]]                                  # pandas Series
            contributions = contributions.append(pd.concat([contrVel,contrHead,contrStd], axis=1))      # pandas DataFrame
        
        # Rename ETMP column to STD (Codar radial case)
        if 'ETMP' in contributions.columns:
            contributions = contributions.rename(columns={"ETMP": "STD"})
        
        # check if there are at least three contributing radial vectors
        if len(contributions.index) >= minContrRads:
            # combine radial contributions to get total vector for the current grid cell
            u, v, C, Cgdop = totalLeastSquare(contributions)
            
            # populate Total Series
            totalData.loc[0] = u                            # VELU
            totalData.loc[1] = v                            # VELV
            totalData.loc[2] = math.sqrt(C[0,0])            # UQAL
            totalData.loc[3] = math.sqrt(C[1,1])            # VQAL
            totalData.loc[4] = C[0,1]                       # CQAL
            totalData.loc[5] = math.sqrt(Cgdop.trace())     # GDOP
            totalData.loc[6] = len(contributions.index)     # NRAD
            
    return totalData


#####################################
# SETUP
#####################################

print('combine_V started')

logger = logging.getLogger(__name__)

desired_width = 320
pd.set_option('display.width', desired_width)
datetime_format = '%Y%m%dT%H%M%SZ'

radialFolder = '/mnt/data/CNR/RADAR/Script/PYTHON/test/test_HFRadarPy/data/radials/ruv/'

timestampPattern = '2021_09_25_0000'

# specify search radius for radial combination (in meters) - FROM WEBFORM
searchRadius = 6000

# specify the lat/lons of the bounding box - FROM WEBFORM
# lonMin, lonMax, latMin, latMax = 7.5, 10.5, 43.25, 44.5                 # HFR-TirLig
# lonMin, lonMax, latMin, latMax = 5.9167, 8.9709, 53.4181, 55.2        # HFR-COSYNA
lonMin, lonMax, latMin, latMax = 13.375, 13.7806, 45.5269, 45.7833      # HFR-NAdr
# lonMin, lonMax, latMin, latMax = -6.1333, -5.0838, 50.2157, 51.0167      # HFR-WHub

# specify grid resoultion in meters - FROM WEBFORM
# gridResolution = 2000       # HFR-TirLig
# gridResolution = 2000       # HFR-COSYNA
gridResolution = 1500       # HFR-NAdr
#gridResolution = 1000       # HFR-WHub

# specify top left point coordinates - FROM WERA FILES
# topLeftLon = 5.916667       # HFR-COSYNA
# topLeftLat = 55.2           # HFR-COSYNA
topLeftLon = 13.375           # HFR-NAdr
topLeftLat = 45.78333         # HFR-NAdr

# specify number of x and y cells and cell size - FROM WERA FILES
# cellSize = 2.000            # HFR-COSYNA
# nx = 100                    # HFR-COSYNA
# ny = 100                    # HFR-COSYNA
cellSize = 1.500            # HFR-NAdr
nx = 22                     # HFR-NAdr
ny = 20                     # HFR-NAdr

# specify if WERA method for creating the geographical grid has to be used
weraGrid = True


#####################################
# RECURSIVELY LIST AND LOAD RADIALS
#####################################

# create list with contributing radials
radialList = []
for filename in glob.iglob(radialFolder + '*/*' + timestampPattern + '*.ruv', recursive=True):
    radialList.append(Radial(filename))
    
# Create DataFrame containing input Radials. The list is only for testing purposes
radialDF = pd.DataFrame(columns=['Radial', 'TBD1', 'TBD2'])
for rad in radialList:
    r = {'Radial':rad, 'TBD1':'tbd1', 'TBD2':'tbd2'}
    thisRadial = pd.DataFrame(data=r, index=[rad.metadata['Site']])
    radialDF = radialDF.append(thisRadial)

# remove empty radials from list
# TO BE DONE MAYBE (IF NECESSARY)

#####################################
# CREATE GEOGRAPHICAL GRID
#####################################

if weraGrid:
    gridGS = createLonLatGridFromBBwera(lonMin, lonMax, latMin, latMax, gridResolution)
    gridGS_2 = createLonLatGridFromTopLeftPointWera(topLeftLon, topLeftLat, cellSize, nx, ny)
else:
    gridGS = createLonLatGridFromBB(lonMin, lonMax, latMin, latMax, gridResolution)


#####################################
# COMBINE RADIALS FOR CREATING TOTAL
#####################################

# create empty total with grid
T = Total(grid=gridGS)

# Fill site_source datframe with contributing radials information
siteNum = 0    # initialization of site number
for Rindex, Rrow in radialDF.iterrows():
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
        thisRadial['AntBearing(NCW)'] = float(rad.metadata['AntennaBearing'].split()[0])
    else:        
        thisRadial['Lat'] = float(rad.metadata['Origin'].split()[0])
        thisRadial['Lon'] = float(rad.metadata['Origin'].split()[1])
        thisRadial['Coverage(s)'] = float(rad.metadata['TimeCoverage'].split()[0])
        thisRadial['RngStep(km)'] = float(rad.metadata['RangeResolutionKMeters'].split()[0])
        thisRadial['Pattern'] = rad.metadata['PatternType'].split()[0]
        thisRadial['AntBearing(NCW)'] = float(rad.metadata['TrueNorth'].split()[0])
    T.site_source = T.site_source.append(thisRadial)
    
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!TO BE COMPLETED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Insert time
# T.time = dt.datetime(xxx)   #to be taken from input timestamp - IT HAS TO BE datetime OBJECT

# fill Total with some metadata
T.metadata['TimeZone'] = rad.metadata['TimeZone']   # trust all radials have the same, pick from the last radial
# T.metadata['Manufacturer'] = xxx                  # to be taken from input dataframe coming from database
T.metadata['AveragingRadius'] = str(searchRadius/1000) + ' km'
T.metadata['GridAxisOrientation'] = '0.0 DegNCW'
T.metadata['GridSpacing'] = str(gridResolution/1000) + ' km'
# T.metadata['TimeStamp'] = xxx                     # to be taken from input dataframe coming from database

# Create Geod object according to the Total CRS
g = Geod(ellps=T.metadata['GreatCircle'].split()[0])

# Create DataFrame for storing indices of radial bins falling within the search radius of each grid cell
combineRadBins = pd.DataFrame(columns=range(len(T.data.index)))

# Figure out which radial bins are within the spatthresh of each grid cell
startGeod = time.time()
for Rindex, Rrow in radialDF.iterrows():
    rad = Rrow['Radial']         
    thisRadBins = T.data.loc[:,['LOND','LATD']].apply(lambda x: radBinsInSearchRadius(x,rad,searchRadius,g),axis=1)
    combineRadBins.loc[Rindex] = thisRadBins

endGeod = time.time()
timeGeod = endGeod-startGeod

print('time_geod_V: ' + str(timeGeod) + ' s')

# Loop over grid points and pull out contributing radial vectors
startCombine = time.time()
combineRadBins = combineRadBins.T
totData = combineRadBins.apply(lambda x: makeTotalVector(x,radialDF), axis=1)

# Assign column names to the combination DataFrame
totData.columns = ['VELU', 'VELV','UQAL','VQAL','CQAL','GDOP','NRAD']

# fill Total with combination results
T.data.loc[:,['VELU', 'VELV','UQAL','VQAL','CQAL','GDOP','NRAD']] = totData
        
endCombine = time.time()
timeCombine = endCombine-startCombine

print('time_combine_V: ' + str(timeCombine) + ' s')

T.data.to_csv('/mnt/data/CNR/RADAR/Script/PYTHON/test/test_combine/combineV.csv')