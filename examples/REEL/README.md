<img src="assets/reel_mock.png">

## Introduction 

Recommenders Engine Example Layout (REEL) is a movie recommendation cross-platform mobile application that demonstrates the integration of some of recommendation algorithms on Microsoft/Recommenders into a mobile application workflow.
The following sections will demonstrate how to locally build REEL. These sections will guide the reader into:
	
* Creating an ML model endpoint
* Setting up and populating [Azure SQL Database](https://docs.microsoft.com/en-us/azure/sql-database/)
* Setting up [Azure Search](https://docs.microsoft.com/en-us/azure/search/)
* Running the app's backend on Azure using [AKS](https://docs.microsoft.com/en-us/azure/aks/)
* Installing the [Xamarin.Forms](https://docs.microsoft.com/en-us/xamarin/xamarin-forms/) app for the mobile client
	
Currently, REEL runs *Simple Algorithm for Recommendation (SAR)* and *LightGBM algorithms*. The application is built using [Xamarin.Forms](https://docs.microsoft.com/en-us/xamarin/xamarin-forms/),
 so it is supported on **iOS**, **Android** and **UWP**. The algorithms are trained on the [MovieLens dataset](https://grouplens.org/datasets/movielens/).




## System Architecture
 
 <img src="assets/system_arch_diagram.png">


# Getting Started

At a high level, this project is composed of 3 main folders
* [backend](backend/README.md): where the Flask web service logic / deployment files are kept. This folder's README also contains instructions for setting up the backend and the database
* [data](data/DATABASE_README.md): where scripts for setting up the SQL database with the movielens dataset are kept
* [mobile](mobile/README.md): where the Xamarin.Forms cross platform application sits. This folder's README contains a high level tour of how the application is structured, as well as information about how to run the application

## Installation process

To get started running the mobile application, a few steps must be followed: 
1. Deploy a SAR model by running the SAR notebook
2. Deploy a LightGBM model by running the LightGBM notebook
3. Set up and deploy the backend to AKS
4. Set up and run the Xamarin.Forms application on your choice of Android, iOS, or UWP

# Demos

## Onboarding

<img src="assets/Onboarding1.gif" width="300px"> 
<img src="assets/Onboarding2.gif" width="300px">

## Browse

<img src="assets/Browse.gif" width="300px">

## Favorites

<img src="assets/Favorites.gif" width="300px">

## Settings

<img src="assets/Settings.gif" width="300px">

## Search

<img src="assets/Search.gif" width="300px">

# 🤝 Contributing

Contributions, issues and feature requests are welcome!<br />Feel free to check [issues page](https://github.com/microsoft/recommenders/issues).

## Show your support

Give a ⭐️ if this project helped you!