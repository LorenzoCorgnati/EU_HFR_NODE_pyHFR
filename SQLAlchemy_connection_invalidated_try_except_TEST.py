#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 19:00:39 2022

@author: lorenz
"""

try:
    # Set and execute the query for retrieving sttion_id, last_data_available and radial_delay
    radialInputSelectQuery = "SELECT station_id, MAX(datetime) as last_data_available, " \
                             "TIMESTAMPDIFF(HOUR, '" + regDate + "', MAX(datetime)) AS " \
                             "radial_delay FROM radial_input_tb WHERE datetime <= '" \
                             + regDate + "' GROUP BY station_id"                   
    delayData = pd.read_sql(radialInputSelectQuery, con=eng)
    # Add registration_date and creation_date to the dataframe
    delayData.loc[:,'registration_date'] = regDate
    delayData.loc[:,'creation_date'] = execDate
    
    # Write the dataframe to the radial_delay_tb table
    delayData.to_sql('radial_delay_tb', con=eng, if_exists='append', index=False, index_label=delayData.columns)
            
    logger.info('radial_delay_tb table succesfully updated for registration date ' + regDate)
    
except sqlalchemy.exc.DBAPIError as err:
    if err.connection_invalidated:
        # if this connection is a "disconnect" condition, run the same query again:
        # the connection will re-validate itself and establish a new connection.
        # The disconnect detection here also causes the whole connection pool to be 
        # invalidated so that all stale connections are discarded.
        delayData = pd.read_sql(radialInputSelectQuery, con=eng)
        # Add registration_date and creation_date to the dataframe
        delayData.loc[:,'registration_date'] = regDate
        delayData.loc[:,'creation_date'] = execDate
        
        # Write the dataframe to the radial_delay_tb table
        delayData.to_sql('radial_delay_tb', con=eng, if_exists='append', index=False, index_label=delayData.columns)
        
        logger.info('radial_delay_tb table succesfully updated for registration date ' + regDate)
    else:
        EHNerr = True
        logger.error('MySQL error ' + err._message())
        logger.info('Exited with errors.')
        sys.exit()