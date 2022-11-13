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


logger = logging.getLogger(__name__)

desired_width = 320
pd.set_option('display.width', desired_width)
datetime_format = '%Y%m%dT%H%M%SZ'

#####################################
# TEST LOAD TOTAL
#####################################

# total_file = '../test/test_HFRadarPy/data/totals/tuv/TOTL_GALI_TEST_2015_06_22_2100.tuv'

# T = Total(total_file)

# empty_T = Total()

# empty_T2 = Total(total_file, empty_total=True)

# total_file_WERA = '../test/test_HFRadarPy/data/totals/cur/20202870719_ico_TEST.cur_asc'
total_file_WERA = '../test/test_HFRadarPy/data/totals/cur/20222432200_got.cur_asc'

TW = Total(total_file_WERA)

# empty_TW = Total(total_file_WERA, empty_total=True)


# #####################################
# # TEST LOAD RADIAL
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



#####################################