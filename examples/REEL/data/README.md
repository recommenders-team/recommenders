## SQL Database and Server Setup on Azure, August 2019 ##

Creating the Database and SQL Server
====================================
1) Go to [portal.azure.com](portal.azure.com) and sign in to your Azure account. If you don't have an account, please click Create Account and follow that process.
2) Create an SQL Database and Server by clicking "+ Create a Resource" and selecting SQL Database.
3) Provide your subscription, resource group, and database name.
4) One of the properties is creating an SQL Server. Click "Create SQL Server". Choose a location of your server that encompasses your location. 
5) Create an admin username and password. Be sure to save the login credientals for the SQL server as they are important for future steps.
6) After creating a SQL Server, properties the database by clicking "Configure Database". For the most cost-effective settings, use "General Purpose", "Serverless" for the Compute Tier, "Gen5" for the Computer Generation, 1 vCore for Max Cores, 0.5 vCores for Min Cores, and 10GB Data max size. Feel free to use Provisioned for the Computer Tier, as it's faster, however, the cost will be higher.


Setting up the Database
=======================

1) Navigate to the SQL Server and select Access Control. Click the "+Add" button, select Add Role Assignment.
2) Select Owner as the role, Assign access to should be set to "Azure AD user, group, or service prinicipal", then click save to add yourself as an Owner so you have admin privilages.
3) Go to the overview tab on your database, click on Firewall Settings and add your current IP so that you can connect to the database and bypass the firewall. **PLEASE NOTE:** If your IP changes, you will not be able to access the database. You must do this step again by adding your current IP to bypass the firewall.

Connecting to the Database
=
Here, you will set up the connection information for properties.py file that will allow local scripts to run and populate the database. 
1) In any file editor, open up properties.py and fill in the neccessary infomation to the correct varaibles inside the double quotes:
	1. The "server" variable is the server URL can be accessed through going to the SQL Server Overview page and coping the text where it says "Server name".
	2. The "database" parameter is the name of the database.
	3. The "username" parameter is the admin username of your SQL Server.
	4. The "password" parameter is the password corrosponding with your username. 

Downloading/Creating the data files
=
**IMPORTANT NOTE:**  Our application uses the ml-1m.zip dataset, however in the download, there isn't a file that contains tmdbID's. The tmdbID's are identifiers for the ML models that points towards specific movies which is critical for the algorithm to work. In order to get these ID's, we must download the 20 million dataset ("ml-20m.zip") and run a script to generate a file that contains the tmdbID's for our movies. The script looks for movies in the 1 million dataset and finds the corrosponding tmdbID's in the 20 million dataset. It then adds these ID's to a new file which is what we'll use to populate the database. It also creates another file called new_movies.csv that will be used to populate the database with movies.

1) Download the 1 million dataset by [clicking here](https://grouplens.org/datasets/movielens/1m/) and clicking on the "ml-1m.zip".
2) Download the 20 million dataset by [clicking here](https://grouplens.org/datasets/movielens/) and under the title "recommended for new research", click on "ml-20m.zip".
3) In the 1 million dataset, extract the movies.dat file and put it into this directory.
4) In the 20 million dataset, extract the links.csv file and put it into this directory. 
5) Run the id_converter.py by double-clicking the file while making sure the movies.dat and the links.csv file are in the same directory as id_converter. If you don't see the new files, open up your console, navigate into your directory, and execute:
``` 
python id_converter.py
```
6) This will create 2 new files, "new_movies.csv" as well as "new_links.csv". These will be used to populate the database.

Populating the Database (Movies)
=
1) Install the latest version of python by clicking [here](https://www.python.org/downloads/)
2) Install Pip by clicking [here](https://pip.pypa.io/en/stable/installing/)
3) If you don't have pyodbc installed, in your console, execute the following command:
```
pip install pyodbc
```
4) Make sure that the populateMovieData.py file is in the same directory as the new_movies.csv file.
5) Run the populateMovieData.py by double-clicking on the file. Again, if it doesn't work, open up your console, navigate into the directory, and then execute:
```
python populateMovieData.py
```

Populating the Database (tmdbID)
=

1) If you haven't installed pyodbc yet through pip using python, please do steps 1-3 inside Populating the Database above. 
2) Make sure new_links.csv is in the same directory as tmdbIDPopulation.py. If so, run tmdbIDPopulation.py by double-clicking on the file. Again, if it doesn't work, open up your console, navigate into the directory, and then execute:
```
python tmdbIDPopulation.py
```



Extra notes:
- There are movie titles that have non-English characters. Please use the utf-8 character set.







