using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Threading.Tasks;
using Xamarin.Forms;
using Xamarin.Forms.Xaml;

using RecommendersDemo.Models;

namespace RecommendersDemo.Views
{
    // Learn more about making custom code visible in the Xamarin.Forms previewer
    // by visiting https://aka.ms/xamarinforms-previewer
    [DesignTimeVisible(false)]
    public partial class MainPage : TabbedPage
    {
        public MainPage()
        {
            InitializeComponent();
            NavigationPage browseNavigationPage;           
            if(Device.RuntimePlatform == Device.UWP || Device.RuntimePlatform == Device.iOS)
            {
                browseNavigationPage = new NavigationPage(new BrowsePage());
            }
            else
            {
                browseNavigationPage = new NavigationPage(new MobileBrowsePage());
            }
            browseNavigationPage.IconImageSource = "home.png";
            browseNavigationPage.Title = "Browse";
            Children.Add(browseNavigationPage);

            var searchNavigationPage = new NavigationPage(new SearchPage());
            searchNavigationPage.IconImageSource = "search.png";
            searchNavigationPage.Title = "Search";
            Children.Add(searchNavigationPage);

            var favoritesNavigationPage = new NavigationPage(new UserMoviePreferencesPage());
            favoritesNavigationPage.IconImageSource = "favorites.png";
            favoritesNavigationPage.Title = "Favorites";
            Children.Add(favoritesNavigationPage);
        }
    }
}