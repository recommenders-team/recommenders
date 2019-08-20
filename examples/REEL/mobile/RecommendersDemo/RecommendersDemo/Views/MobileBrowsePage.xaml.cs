using RecommendersDemo.Models;
using RecommendersDemo.ViewModels;
using System;
using System.ComponentModel;
using Xamarin.Forms;

namespace RecommendersDemo.Views
{
    // Learn more about making custom code visible in the Xamarin.Forms previewer
    // by visiting https://aka.ms/xamarinforms-previewer
    [DesignTimeVisible(false)]
    public partial class MobileBrowsePage : ContentPage
    {
        public BrowseViewModel ViewModel { get; set; }

        public MobileBrowsePage()
        {
            InitializeComponent();
            BindingContext = ViewModel = new BrowseViewModel(async ex => await this.DisplayAlert("Error", ex.Message, "Ok").ConfigureAwait(false), UserMoviePreferences.getInstance(), new Services.RestClient());
        }

        protected override void OnAppearing()
        {
            base.OnAppearing();
            if (ViewModel.RecommendedMovies.Count == 0)
                _ = ViewModel.RefreshRequestRecommendations();
        }

        private async void ViewSettingsButtonClicked(object sender, EventArgs e)
        {
            await Navigation.PushModalAsync(new SettingsPage()).ConfigureAwait(false);
        }

        async void OnItemSelected(object sender, EventArgs e)
        {
            var item = (Xamarin.Forms.StackLayout)sender;
            await Navigation.PushModalAsync(new MovieDetailPage((MovieDetailViewModel)item.BindingContext)).ConfigureAwait(false);
        }
    }
}