using RecommendersDemo.ViewModels;
using System;
using Xamarin.Forms;

namespace RecommendersDemo.Views
{
    public partial class MovieDetailPage : ContentPage
    {
        public MovieDetailPage(MovieDetailViewModel viewModel)
        {
            InitializeComponent();
            BindingContext = viewModel;
        }

        private async void ViewSettingsButtonClicked(object sender, EventArgs e)
        {
            await Navigation.PushModalAsync(new SettingsPage()).ConfigureAwait(false);
        }

        async void OnExitClick(object sender, EventArgs e)
        {
            await Navigation.PopModalAsync().ConfigureAwait(false);
        }

        protected override bool OnBackButtonPressed()
        {
            Navigation.PopModalAsync().ConfigureAwait(false);
            return true;
        }
    }
}