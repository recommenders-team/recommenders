using RecommendersDemo.Models;
using RecommendersDemo.ViewModels;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xamarin.Essentials;
using Xamarin.Forms;
using Xamarin.Forms.Xaml;

namespace RecommendersDemo.Views
{
    [XamlCompilation(XamlCompilationOptions.Compile)]
    public partial class PersonasPage : ContentPage
    {
        PersonasViewModel viewModel;
        UserMoviePreferences preferences = UserMoviePreferences.getInstance();

        public PersonasPage()
        {
            InitializeComponent();
            BindingContext = viewModel = new PersonasViewModel(async ex => await this.DisplayAlert("Error", ex.Message, "Ok").ConfigureAwait(false));
        }

        private async void RedirectToCustomPage(object sender, EventArgs e)
        {
            await Navigation.PushModalAsync(new CustomOnboardInfo()).ConfigureAwait(false);
        }

        private async void RedirectToMainPage(object sender, EventArgs e)
        {
            if (viewModel.chosenPersona != null)
            {
                preferences.AddMultiplePreferences((List<Movie>)viewModel.chosenPersona.getLikedMovies());
                if (App.algorithm == "sar")
                {
                    await Navigation.PushModalAsync(new SarAlgorithmPage()).ConfigureAwait(false);
                }
                else
                {
                    await Navigation.PushModalAsync(new LgbmAlgorithmPage()).ConfigureAwait(false);
                }
            }
        }

        private void SetChosenPersona(object sender, EventArgs e)
        {
            var item = (ImageButton) sender;
            viewModel.SetChosenPersona(item.CommandParameter.ToString());
            viewModel.ToggleCheckmark(item.CommandParameter.ToString());
            viewModel.SetNextButtonBackgroundColor();
        }
    }
}