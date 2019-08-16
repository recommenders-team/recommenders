using System;
using System.ComponentModel;
using Xamarin.Forms;

namespace RecommendersDemo.Views
{
    [DesignTimeVisible(false)]
    public partial class FirstOnboardingPage : ContentPage
    {
        public FirstOnboardingPage()
        {
            InitializeComponent();
        }

        private async void Redirect(object sender, EventArgs e)
        {
            await Navigation.PushModalAsync(new PersonasPage()).ConfigureAwait(false);
        }
    }
}