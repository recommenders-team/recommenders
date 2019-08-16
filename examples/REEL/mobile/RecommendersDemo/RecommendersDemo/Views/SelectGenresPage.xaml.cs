using RecommendersDemo.Models;
using RecommendersDemo.ViewModels;
using System;
using System.ComponentModel;
using System.Runtime.InteropServices.ComTypes;
using Xamarin.Forms;

namespace RecommendersDemo.Views
{
    [DesignTimeVisible(false)]
    public partial class SelectGenresPage : ContentPage
    {
        SelectGenresViewModel ViewModel;
        public SelectGenresPage()
        {
            InitializeComponent();
            BindingContext = ViewModel = new SelectGenresViewModel();
        }

        private async void RedirectToMovieCarousel(object sender, EventArgs e) {
            if (ViewModel.ChangeButtonColor())
            {
                await Navigation.PushModalAsync(new CustomOnboardingPart2()).ConfigureAwait(false);
            }         
        }
    }
}