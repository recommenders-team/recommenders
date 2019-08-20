import config
import pyodbc
import os
import time
import sys

"""
Adds a column to the MOVIES table called tmdbID and populates it with every movies' correct tmdbID
Command to add the new column: ALTER TABLE MOVIES ADD tmdbID int;
"""
def populateTMBDID():

    try:    
        movieFile = open("new_links.csv", mode="r", encoding="utf-8")
    except:
        print("Couldn't find the new_links.csv file, now shutting down...")
        time.sleep(5)
        sys.exit(0)

    movieData = movieFile.readline() # gets rid of the title line

    updatedElements = 0

    # connect to database
    cnxn = pyodbc.connect(
        'DRIVER=' + config.driver + ';SERVER=' + config.server + ';PORT=1433;DATABASE=' + config.database + ';UID=' + config.username + ';PWD=' + config.password)
    cursor = cnxn.cursor()

    # while there is data to read, grab the next line, split the data by commas, and set variables
    while movieFile.readable():
        movieData = movieFile.readline()
        movieDataSplit = movieData.split(",")
        movieID = movieDataSplit[3]
        tmdbID  = movieDataSplit[2]

        # get rid of the \n at the end of the tmdbID
        movieID = movieID[:(len(movieID) - 1)]

        # make the two id's ints instead of doubles
        movieID = (int)(movieID)
        tmdbID  = (int)(tmdbID)

        # Go through each rating and add it to the ratings table
        cursor.execute("UPDATE MOVIES SET tmdbID = ? WHERE movieID = ?", tmdbID, movieID)
        cnxn.commit() # commit the entries into the database

        updatedElements += 1
        os.system('cls')
        print("Updated " + str(updatedElements) + " movies in movie table, on movie " + str(movieID))
    
    print("Finished populating tmdbID's! Now exiting...")
    time.sleep(5)

populateTMBDID()