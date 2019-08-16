using RecommendersDemo.Models;
using RecommendersDemo.ViewModels;
using System;
using System.Collections.Generic;
using System.Globalization;
using Xamarin.Forms;
using Xamarin.Forms.Xaml;

namespace RecommendersDemo.Views
{
    [XamlCompilation(XamlCompilationOptions.Compile)]
    public partial class SettingsPage : ContentPage
    {
        SettingsViewModel viewModel;
        UserMoviePreferences preferences = UserMoviePreferences.getInstance();
        UserPersonas personas = UserPersonas.GetInstance();

        public SettingsPage()
        {
            InitializeComponent();
            BindingContext = viewModel = new SettingsViewModel();
        }

        protected override void OnAppearing()
        {
            base.OnAppearing();
            description.Text = viewModel.GetDescription(App.algorithm.ToUpper(CultureInfo.InvariantCulture));
        }

        void PersonaButtonClicked(object sender, EventArgs e)
        {
            var button = (ImageButton)sender;

            foreach (PersonaWrapper persona in personas.GetAllPersonas())
            {
                if (button.ClassId.Equals(persona.persona.Name, StringComparison.Ordinal))
                {
                    persona.ShowCheckmark = "True";
                }
                else
                {
                    persona.ShowCheckmark = "False";
                }
            }

            preferences.ClearPreferences();
            preferences.AddMultiplePreferences((List<Movie>)((PersonaWrapper)button.BindingContext).persona.getLikedMovies());
        }

        void AlgorithmButtonClicked(object sender, EventArgs e)
        {
            var button = (Button)sender;
            var classId = button.ClassId;
            
            if (classId == "sarButton")
            {
                App.algorithm = "sar";
                description.Text = viewModel.GetDescription("SAR");
                viewModel.SarCheck = "True";
                viewModel.LgbmCheck = "False";
            } else
            {
                App.algorithm = "lgbm";
                description.Text = viewModel.GetDescription("LGBM");
                viewModel.SarCheck = "False";
                viewModel.LgbmCheck = "True";
            }
        }

        async void OnExitClick(object sender, EventArgs e)
        {
            await Navigation.PushModalAsync(new MainPage()).ConfigureAwait(false);
        }

        protected override bool OnBackButtonPressed()
        {
            Navigation.PushModalAsync(new MainPage());
            return true;
        }

        private async void OnCreatePersona(object sender, EventArgs e)
        {
            preferences.ClearPreferences();
            await Navigation.PushModalAsync(new CustomOnboardInfo()).ConfigureAwait(false);
        }
    }
}