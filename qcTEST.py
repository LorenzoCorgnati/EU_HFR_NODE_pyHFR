#!/usr/bin/python3

import datetime as dt
import pandas as pd
import numpy as np
import glob
import io
import logging
import os
import calc
from radials import Radial
from totals import Total

# #####################################
# # TEST RADIAL QC
# #####################################

radial_file = '../test/test_HFRadarPy/data/radials/ruv/TINO/RDLm_TINO_2021_01_08_0800.ruv'

R = Radial(radial_file)

# empty_R = Radial(radial_file, empty_radial=True)

# radial_file_WERA = '../test/test_HFRadarPy/data/radials/crad/AURI/20221311500_sn1.crad_ascii'
radial_file_WERA = '../test/test_HFRadarPy/data/radials/crad/SYLT/2022117072000_syl.CUR.crad_ascii'

RW = Radial(radial_file_WERA)

# empty_RW = Radial(radial_file_WERA, empty_radial=True)

radial_file_LERA = '/mnt/data/CNR/RADAR/DATI/Dati_HFR_BoB/Radials_asc/MIMZ/2022051260700_mimz.crad_ascii'

RL=Radial(radial_file_LERA)

# initialize
R.initialize_qc()
RW.initialize_qc()
RL.initialize_qc()

# VART
RW.qc_ehn_maximum_variance(1)
RL.qc_ehn_maximum_variance(1)
R.qc_ehn_maximum_variance(1)

# OWTR
R.qc_ehn_over_water()
RW.qc_ehn_over_water()
RL.qc_ehn_over_water()

# AVRB
R.qc_ehn_avg_radial_bearing(175,210)
RW.qc_ehn_avg_radial_bearing(175,210)
RL.qc_ehn_avg_radial_bearing(175,210)

# RDCT
R.qc_ehn_radial_count(175)
RW.qc_ehn_radial_count(210)
RL.qc_ehn_radial_count(175)

# CSPD
R.qc_ehn_maximum_velocity(2)
RW.qc_ehn_maximum_velocity(1.7)
RL.qc_ehn_maximum_velocity(1.2)

# MDFL
R.qc_ehn_median_filter(10,0.5)
RW.qc_ehn_median_filter(10,0.5)
RL.qc_ehn_median_filter(10,0.5)





#####################################
# TEST LOAD TOTAL
#####################################

# total_file = '../test/test_HFRadarPy/data/totals/tuv/TOTL_GALI_TEST_2015_06_22_2100.tuv'

# T = Total(total_file)

# empty_T = Total()

# empty_T2 = Total(total_file, empty_total=True)

# total_file_WERA = '../test/test_HFRadarPy/data/totals/cur/20202870719_ico_TEST.cur_asc'

# TW = Total(total_file_WERA)

# empty_TW = Total(total_file_WERA, empty_total=True)



#####################################