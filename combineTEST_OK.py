#!/usr/bin/python3

import datetime as dt
import pandas as pd
import numpy as np
import glob
import io
import logging
import os
from radials import Radial
from totals import Total, createLonLatGrid
from calc import true2mathAngle
from pyproj import Geod
import latlon
import time
import math

def totalLeastSquare(VelHead):
    """
    This function calculates the u/v components of a total vector from 2 to n 
    radial vector components using weighted Least Square method.
    
    INPUTS:
        VelHead: DataFrame containing contributor radial velocities and bearings
        
    OUTPUTS:
        u: U component of the total vector
        v: V component of the total vector
        C: covariance matrix assuming uniform unit errors for all radials (i.e. GDOP)
    """
    # Convert angles from true convention to math convention
    VelHead['HEAD'] = true2mathAngle(VelHead['HEAD'].to_numpy())
    
    # Form the angle matrices DA RIVEDERE DIMENSIONI MATRICE
    A = np.stack((np.array([np.cos(np.deg2rad(VelHead['HEAD']))]),np.array([np.sin(np.deg2rad(VelHead['HEAD']))])),axis=-1)[0,:,:]
    
    # Evaluate the covariance matrix C (variance(U) = C(1,1) and variance(V) = C(2,2))
    C = np.linalg.inv(np.matmul(A.T, A))
    
    # Calculate the u and v for the total vector
    a = np.matmul(C, np.matmul(A.T, VelHead['VELO'].to_numpy()))
    u = a[0]
    v = a[1]    
    
    return u, v, C


#####################################
# SETUP
#####################################

print('combine_NV started')

logger = logging.getLogger(__name__)

desired_width = 320
pd.set_option('display.width', desired_width)
datetime_format = '%Y%m%dT%H%M%SZ'

radialFolder = '/mnt/data/CNR/RADAR/Script/PYTHON/test/test_HFRadarPy/data/radials/ruv/'

timestampPattern = '2021_09_25_0000'

# specify search radius for radial combination (in meters)
searchRadius = 6000

# specify the lat/lons of the bounding box
lonMin, lonMax, latMin, latMax = 7.5, 10.5, 43.25, 44.5

# specify grid resoultion in meters
gridResolution = 2000

# set minimum number of contributing radial sites
minContrSites = 2
# set minimum number of contributing radial vectors
minContrRads = 3


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

gridGS = createLonLatGrid(lonMin, lonMax, latMin, latMax, gridResolution)


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
    thisRadial['Lat'] = float(rad.metadata['Origin'].split()[0])
    thisRadial['Lon'] = float(rad.metadata['Origin'].split()[1])
    thisRadial['Coverage(s)'] = float(rad.metadata['TimeCoverage'].split()[0])
    thisRadial['RngStep(km)'] = float(rad.metadata['RangeResolutionKMeters'].split()[0])
    thisRadial['Pattern'] = rad.metadata['PatternType'].split()[0]
    thisRadial['AntBearing(NCW)'] = float(rad.metadata['AntennaBearing'].split()[0])
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

# Make search radilus a list to vary for each grid cell
sR = len(T.data.index)*[searchRadius]

# Create DataFrame for storing indices of radial bins falling within the search radius of each grid cell
combineRadBins = pd.DataFrame(columns=range(len(T.data.index)))

# Figure out which radial grid points are within the spatthresh of each grid cell
startGeod = time.time()
for Rindex, Rrow in radialDF.iterrows():
    rad = Rrow['Radial']
    # set index name as the site code in the combineRadBins DataFrame
    thisRadBins = pd.DataFrame(index=[Rindex],columns=range(len(T.data.index)))
    for k in T.data.index:
        az12,az21,cellToRadDist = g.inv(len(rad.data.LOND)*[T.data.LOND[k]],len(rad.data.LATD)*[T.data.LATD[k]],rad.data.LOND,rad.data.LATD)
        # thisRadBins[k][0] = np.where(cellToRadDist < sR[k])[0].tolist()
        thisRadBins.loc[Rindex,k] = np.where(cellToRadDist < sR[k])[0].tolist()
        
    combineRadBins = combineRadBins.append(thisRadBins)

endGeod = time.time()
timeGeod = endGeod-startGeod

print('time_geod_NV: ' + str(timeGeod) + ' s')

# Loop over grid points and pull out contributing radial vectors
startCombine = time.time()
for k in combineRadBins.columns:
    thisCell = combineRadBins[k]
    contrRad = thisCell[thisCell.str.len() != 0]
    # check if there are at least two contributing radial sites
    if contrRad.size >= minContrSites:
        # loop over contributing radial indices for collecting velocities and angles
        contributions = pd.DataFrame(columns=['VELO', 'HEAD'])
        for idx in contrRad.index:
            contrVel = radialDF.loc[idx]['Radial'].data.VELO[contrRad[idx]]                 # pandas Series
            contrHead = radialDF.loc[idx]['Radial'].data.HEAD[contrRad[idx]]                # pandas Series
            contributions = contributions.append(pd.concat([contrVel,contrHead], axis=1))   # pandas DataFrame
        
        # check if there are at least three contributing radial vectors
        if len(contributions.index) >= minContrRads:
            # combine radial contributions to get total vector for the current grid cell
            u, v, C = totalLeastSquare(contributions)
        
            # populate Total object
            T.data.loc[k,'VELU'] = u
            T.data.loc[k,'VELV'] = v
            T.data.loc[k,'UQAL'] = math.sqrt(C[0,0])
            T.data.loc[k,'VQAL'] = math.sqrt(C[1,1])
            T.data.loc[k,'CQAL'] = C[0,1]
            T.data.loc[k,'NRAD'] = len(contributions.index)
        
        # STUDIARE GDOPMaxOrthog DA PAPER GURGEL E AGGIUNGERE COLONNA 'GDOP' A T.data
        
endCombine = time.time()
timeCombine = endCombine-startCombine

print('time_combine_NV: ' + str(timeCombine) + ' s')

T.data.to_csv('/mnt/data/CNR/RADAR/Script/PYTHON/test/test_combine/combineNV.csv')