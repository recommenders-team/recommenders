import pyodbc
import os
import config
import time
import sys

"""
Populates the database with information about each movie such as there movieID, title, date, and genres
Reads this information through a dataset downloaded from MovieLens
Data Format:
    1) COUNT,MOVIE_ID,MOVIE_TITLE MOVIE_YEAR,GENRE|GENRE|GENRE
    2) COUNT,MOVIE_ID,"MOVIE_TITLE MOVIE_YEAR",GENRE|GENRE|GENRE - there is a comma and/or parenthesis
"""
def populateMovieData():

    COMMIT_NUMBER = 1000 # after every COMMIT_NUMBER of insert statements, the program will commit those inserts to the database - used for optimization

    showProgress = True  # this will display the number of movies that go into the database HOWEVER slows down insertion process    

    try:    
        movieFile = open("new_movies.csv", mode="r", encoding="utf-8")
    except:
        print("Couldn't find the new_movies.csv file, now shutting down...")
        time.sleep(5)
        sys.exit(0)
    print("Parsing info...\n")

    movieList = [] # container holding all the lists of movies
    try:
        while movieFile.readable():
            # reset all the variables and read the next line
            movieData   = movieFile.readline()
            movieID     = ""
            movieTitle  = ""
            movieYear   = ""
            movieGenres = []

            # gets rid of the count column (the first column)
            movieData = movieData[movieData.index(",") + 1:]

            # MOVIE ID
            # finds where the first comma is and substrings from the beginning to that point to find movie ID
            # then removes the movie ID and the first comma from the movieData string
            movieID   = movieData[:movieData.index(",")]
            movieData = movieData[movieData.index(",") + 1:]

            # MOVIE TITLE AND YEAR
            # movieString is a string that holds the movieTitle and the movieYear: 'Toy Story (2006)'
            # if the first character in movieData is a double quote, that means there is a comma in the movieTitle so find movieString through the end quote instead of commas
            #   Example: "Example, The (2005)", Genres
            # if the first character isn't a double quote, then there's no comma in the title so its safe to look for the next comma
            #   Example: Toy Story (2006), Genres

            if movieData[0] == '"':
                rightQuoteIndex = movieData[1:].index('"') # go from back to front
                movieString     = movieData[1:(rightQuoteIndex + 1)]
                movieData       = movieData[(rightQuoteIndex + 3):]
            else:
                movieString = movieData[:movieData.index(",")]
                movieData   = movieData[movieData.index(",") + 1:]

            # because the years are always in a 4 digit format, the end of the movieTitle and beginning of the movieYear can always be found in the same format
            movieYear   = movieString[(len(movieString) - 5):(len(movieString) - 1)]
            movieTitle  = movieString[:(len(movieString) - 7)]

            # MOVIE GENRES
            movieGenres = movieData
            
            # get's rid of the \n that's at the end of the last listed genre
            movieGenres = movieGenres[:(len(movieGenres) - 1)]
            
            # print results for debugging purposes
            '''print("ID:     " + movieID)
            print("Title:  " + movieTitle)
            print("Year:   " + movieYear)
            print("Genres: " + str(movieGenres))
            print()'''

            # add it to the list of movies
            movie = [movieID, movieTitle, movieYear, movieGenres]
            movieList.append(movie)
    
    except ValueError:
        pass # codes throws this error once it finishes reading for some reason
    finally:
        print("Finished parsing data successfully")
        movieFile.close()


    print("Connecting to the database and creating table...")
    # Setting up info for program to connect to the database
    # Connecting to the database with the information provided above
    cnxn = pyodbc.connect(
        'DRIVER=' + config.driver + ';SERVER=' + config.server + ';PORT=1433;DATABASE=' + config.database + ';UID=' + config.username + ';PWD=' + config.password)
    cursor = cnxn.cursor()

    # Drop if the table exists, then create a table for movies - CURRENTLY DONT HAVE PERMISSIONS TO DO SO

    command = 'DROP TABLE IF EXISTS MOVIES'
    cursor.execute(command)
    cnxn.commit()

    # Create the table for movies - CURRENTLY DONT HAVE PERMISSIONS TO DO SO
    command = 'CREATE TABLE MOVIES(movieID INT, movieTitle VARCHAR(1000), movieYear CHAR(4), movieGenres VARCHAR(1000), tmdbID INT, CONSTRAINT MOVIES_pk PRIMARY KEY (movieID));'
    cursor.execute(command)
    cnxn.commit()

    # Go through each movie and add it to the movie table
    movieNumber = 1

    print("Successfully made tables")

    print("Starting to populate the database...")
    
    for movie in movieList:
        cursor.execute("INSERT INTO MOVIES (movieID, movieTitle, movieYear, movieGenres) VALUES(?, ?, ?, ?)", movie[0], movie[1], movie[2], movie[3])
        
        if movieNumber % COMMIT_NUMBER == 0:
            cnxn.commit()
        
        if showProgress:
            os.system('cls')
            print("Put " + str(movieNumber) + " movies into the database")
        
        movieNumber += 1

    cnxn.commit() # commit the last >COMMIT_NUMBER entries into the database

    print("Successfully inserted all the data into the database")
    time.sleep(5)
    

populateMovieData()