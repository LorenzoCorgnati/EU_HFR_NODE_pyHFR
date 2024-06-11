#!/usr/bin/python3


# Created on Sat Mar 23 19:16:46 2024

# @author: Lorenzo Corgnati
# e-mail: lorenzo.corgnati@sp.ismar.cnr.it


# This application synchronizes the native radial and total files sent by the providers
# from the access virtual machine to the network folders inside the EU_HFR_NODE mount
# where the EU_HFR_NODE processing applications work. 

# The application retrieves data from the home folder of each provider's user on the 
# access virtual machine and saves them on the corresponding network folder on the 
# EU_HFR_NODE mount.

import os
import sys
import paramiko
import getopt
import logging
import re

####################
# MAIN DEFINITION
####################

def main(argv):
    
#####
# Setup
#####
       
    # Set the argument structure
    try:
        opts, args = getopt.getopt(argv,"h",["help"])
    except getopt.GetoptError:
        print('Usage: syncDataFromAccessVM.py')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('Usage: syncDataFromAccessVM.py')
            sys.exit()
            
    # Create logger
    logger = logging.getLogger('EU_HFR_NODE_NRT_synchDataFromAccessVM')
    logger.setLevel(logging.INFO)
    # Create logfile handler
    logFilename = '/var/log/EU_HFR_NODE_NRT/EU_HFR_NODE_NRT_synchDataFromAccessVM.log'
    lfh = logging.FileHandler(logFilename)
    lfh.setLevel(logging.INFO)
    # Create formatter
    formatter = logging.Formatter('[%(asctime)s] -- %(levelname)s -- %(module)s - %(message)s', datefmt = '%d-%m-%Y %H:%M:%S')
    # Add formatter to lfh and ch
    lfh.setFormatter(formatter)
    # Add lfh and ch to logger
    logger.addHandler(lfh)
    
    # Set parameters for ssh connection
    sshConfig = {
      'hostname': '150.145.136.106',
      'alias': 'access.hfrnode.eu',
      'username': 'accessadmin',
      'password': '',
    }
    
    # Initialize error flag
    sDAerr = False
    
    logger.info('Synchronization started.')
    
    try:
        
#####
# Set the source and destination folder path patterns
#####
        
        # Set the destination base folder
        destBaseFolder = '/home/radarcombine/EU_HFR_NODE/'
        
        # Set the source base folder pattern
        srcBasePattern = '/home/hfr_*'
        
        # Set the source folder path pattern
        srcFolderPattern = os.path.join(srcBasePattern, 'NRT_HFR_DATA', 'HFR-*')
    
#####
# Establish ssh connection to access VM
#####

        # Initialize the SSH client
        client = paramiko.SSHClient()

        # Add to known hosts
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        # Connect
        try:
            client.connect(hostname=sshConfig['alias'], username=sshConfig['username'])
            logger.info('SSH connection established with the remote server')
        except Exception as err:
            try:
                client.connect(hostname=sshConfig['hostname'], username=sshConfig['username'])
                logger.info('SSH connection established with the remote server')
            except Exception as err:
                logger.info('Cannot connect to the SSH server')
                exit()
            
#####
# Retrieve the source folders for each network
#####

        # Set the command for listing the source folders
        findSrcFolderCommand = 'find ' + srcBasePattern + ' -type d -wholename \"' + srcFolderPattern + '\" 2>&1 | grep -v \"Permission denied\"'
            
        # Execute the command for listing the source folders
        try:
            stdin, stdout, stderr = client.exec_command(findSrcFolderCommand)
            err = stderr.read().decode()
            if err:
                logger.info(err)    
                exit(1)
            allSrcFolders = stdout.read().decode().split('\n')
            logger.info('Source folders succesfully lsited from remote server')
            
        finally:
            client.close()
        
#####
# Execute synchronization
#####

        if allSrcFolders:
            # Manage source folder paths: keep only folder paths containing 'Radials' and 'Totals' strings
            allSrcFolders = [item for item in allSrcFolders if (('Radials' in item) or ('Totals' in item))]
            
            # Manage source folder paths: keep only folder paths that are not subpaths of others
            srcFolders = []
            for srcFolder in allSrcFolders:
                res = [item for item in allSrcFolders if srcFolder in item]
                if len(res) == 1:
                    srcFolders.append(srcFolder)
                    
            # Prepare the regular expression pattern for building the destination path
            reSearchPattern = os.path.join(srcBasePattern, 'NRT_HFR_DATA/').replace('*','.+')
            
            # Synchronize files
            for srcFolder in srcFolders:
                # Get the source base path
                srcBaseFolder = str(re.compile(reSearchPattern).search(srcFolder).group(0))
                
                # Build destination folder path
                destFolder = srcFolder.replace(srcBaseFolder, destBaseFolder)
                    
                # Execute rsync commands
                os.system('rsync -rltvz ' + sshConfig['username'] + '@' + sshConfig['alias'] +':' + os.path.join(srcFolder,'') + ' ' + os.path.join(destFolder,'') + ' --log-file=' + logFilename)
    
    except Exception as err:
        sDAerr = True
        logger.error(err.args[0])    
    
    
####################
    
    if(not sDAerr):
        logger.info('Successfully executed.')
    else:
        logger.error('Exited with errors.')
            
####################


#####################################
# SCRIPT LAUNCHER
#####################################    
    
if __name__ == '__main__':
    main(sys.argv[1:])
    
    