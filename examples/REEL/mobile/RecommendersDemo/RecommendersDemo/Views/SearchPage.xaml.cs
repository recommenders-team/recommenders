using RecommendersDemo.ViewModels;
using System;
using System.ComponentModel;
using System.Globalization;
using Xamarin.Forms;

namespace RecommendersDemo.Views
{
    [DesignTimeVisible(false)]
    public partial class SearchPage : ContentPage
    {
        SearchViewModel viewModel;

        public SearchPage()
        {
            InitializeComponent();
            BindingContext = viewModel = new SearchViewModel(async ex => await this.DisplayAlert("Error", ex.Message, "Ok").ConfigureAwait(false), new Services.RestClient());
        }

        private async void ViewSettingsButtonClicked(object sender, EventArgs e)
        {
            await Navigation.PushModalAsync(new SettingsPage()).ConfigureAwait(false);
        }

        private void OnSearchClick(object sender, EventArgs e)
        {
            var item = (Xamarin.Forms.SearchBar)sender;
            viewModel.GetSearchResults(userMovieQuery_entry.Text.ToString(CultureInfo.InvariantCulture), "1"); // 1 is the first 5 movies of the search result
        }

        async void OnItemSelected(object sender, EventArgs e)
        {
            var item = (Xamarin.Forms.StackLayout)sender;
            await Navigation.PushModalAsync(new MovieDetailPage((MovieDetailViewModel)item.BindingContext)).ConfigureAwait(false);
        }
    }
}