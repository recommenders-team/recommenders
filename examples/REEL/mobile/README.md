# Introduction
This mobile application was built using Xamarin.Forms which is an open source framework for building iOS, Android, and Windows apps. Learn more about Xamarin Forms platform at https://www.xamarin.com/forms.

# Installation Guide
Pre-requisite: the backend web service is either running locally on your computer, or is deployed to AKS

1. Ensure your version of Visual Studio has workspace for developing Xamarin.Forms applications. If it does not, launch the Visual Studio Installer and install the Mobile development with .NET workload (https://docs.microsoft.com/en-us/xamarin/get-started/installation) 
2. Launch Visual Studio and open the solution (.sln) file within the mobile folder
3. Update the Resources.resx file to contain the proper URLs of the relevant endpoints.
    * Note: These URLs can be pointing at the backend service running locally on the computer, but you will only be able to run the application in debug mode. You will not be able to successfully package and distribute a fully-functional application without deploying the backend to AKS
4. Run the application on the platform of your choice
    * Note: To run the application as a Windows app, you will need to put your Windows device in developer mode

# Project Navigation

The folder `recommenders_demo` is where all of the non platform-specific C# logic and XAML styling is put. This project was built to follow the MVVM design pattern:

* Models: Where model objects are stored
* Services: Contains the RestClient class, which is responsible for communicating with the Flask web service backend 
* ViewModels: Where more specific business logic for modifying the data in each page is. For every page (ex: BrowsePage.xaml), there is a corresponding ViewModel (ex: BrowseViewModel.cs)
* Views: Where the pages of the application (.xaml) and their respective codebehind (.xaml.cs) files are stored

The folders `recommenders_demo.Android`, `recommenders_demo.iOS`, and `recommenders_demo.UWP` each contain more platform specific resources (like images/icons) and minimal logic 

To folder `UnitTests` are where our unit tests sit for all relevant Model and ViewModel objects. If you contribute additional functionality, please be sure to add appropriate unit tests.