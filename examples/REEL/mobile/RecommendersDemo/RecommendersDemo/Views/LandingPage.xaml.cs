using System;
using System.ComponentModel;
using Xamarin.Forms;

namespace RecommendersDemo.Views
{
    [DesignTimeVisible(false)]
    public partial class LandingPage : ContentPage
    {
        public LandingPage()
        {
            InitializeComponent();
        }

        private async void Redirect(object sender, EventArgs e)
        {
            await Navigation.PushModalAsync(new FirstOnboardingPage()).ConfigureAwait(false);
        }
    }
}