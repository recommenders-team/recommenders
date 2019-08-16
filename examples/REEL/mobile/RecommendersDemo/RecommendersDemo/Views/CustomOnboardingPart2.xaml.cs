using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using Xamarin.Forms;
using Xamarin.Forms.Xaml;

namespace RecommendersDemo.Views
{
    [DesignTimeVisible(false)]
    public partial class CustomOnboardingPart2 : ContentPage
    {
        public CustomOnboardingPart2()
        {
            InitializeComponent();
        }

        private async void Redirect(object sender, EventArgs e)
        {
            await Navigation.PushModalAsync(new OnboardingMovieSelectionPage()).ConfigureAwait(false);
        }
    }
}