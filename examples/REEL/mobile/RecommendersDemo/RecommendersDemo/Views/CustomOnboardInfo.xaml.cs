using RecommendersDemo.Models;
using RecommendersDemo.ViewModels;
using System;
using System.ComponentModel;
using System.Runtime.InteropServices.ComTypes;
using Xamarin.Forms;

namespace RecommendersDemo.Views
{
    [DesignTimeVisible(false)]
    public partial class CustomOnboardInfo: ContentPage
    {
        public CustomOnboardInfo()
        {
            InitializeComponent();
        }

        private async void Redirect(object sender, EventArgs e)
        {
            await Navigation.PushModalAsync(new SelectGenresPage()).ConfigureAwait(false);
        }
    }
}