using RecommendersDemo.Models;
using RecommendersDemo.ViewModels;
using System;
using System.ComponentModel;

using Xamarin.Forms;

namespace RecommendersDemo.Views
{
    [DesignTimeVisible(false)]
    public partial class OnboardingMovieSelectionPage : ContentPage
    {
        OnboardingMovieSelectionViewModel ViewModel;
        public OnboardingMovieSelectionPage()
        {
            InitializeComponent();
            BindingContext = ViewModel = new OnboardingMovieSelectionViewModel(UserMoviePreferences.getInstance(), new Services.RestClient());
            var prevItem = new ToolbarItem
            {
                Text = "**Prev**",
                IconImageSource = "prev",
                CommandParameter = false
            };
            prevItem.SetBinding(MenuItem.CommandProperty, nameof(ViewModel.PanPositionChangedCommand));

            var nextItem = new ToolbarItem
            {
                Text = "**Next**",
                IconImageSource = "next",
                CommandParameter = true
            };
            nextItem.SetBinding(MenuItem.CommandProperty, nameof(ViewModel.PanPositionChangedCommand));

            ToolbarItems.Add(prevItem);
            ToolbarItems.Add(nextItem);
        }

        private void checkValid(object sender, EventArgs e)
        {
            ViewModel.ChangeButtonColor();
        }

        private async void Redirect(object sender, EventArgs e)
        {
            if (ViewModel.ChangeButtonColor())
            await Navigation.PushModalAsync(new AlgorithmPage()).ConfigureAwait(false);
        }
    }
}