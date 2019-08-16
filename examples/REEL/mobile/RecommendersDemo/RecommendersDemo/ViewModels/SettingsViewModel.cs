using RecommendersDemo.Models;
using RecommendersDemo.Properties;
using RecommendersDemo.Services;
using System;
using System.Collections.Generic;
using Xamarin.Essentials;
using Xamarin.Forms;

namespace RecommendersDemo.ViewModels
{
    public class SettingsViewModel : BaseViewModel
    {
        private Dictionary<string, string> algorithmDescription = new Dictionary<string, string>
        {
            { "SAR", Resources.SarDescription },
            { "LGBM", Resources.LgbmDescription }
        };

        private readonly UserPersonas userPersonas;

        private List<PersonaWrapper> personas = new List<PersonaWrapper>();
        public Command OpenGitHubLink { get; }
        public List<PersonaWrapper> Personas
        {
            get { return personas; }
            set { SetProperty(ref personas, value); }
        }

        private string sarCheck;
        public string SarCheck
        {
            get { return sarCheck; }
            set { SetProperty(ref sarCheck, value);  }
        }

        private string lgbmCheck;
        public string LgbmCheck
        {
            get { return lgbmCheck; }
            set { SetProperty(ref lgbmCheck, value); }
        }

        public SettingsViewModel()
        {
            userPersonas = UserPersonas.GetInstance();
            personas = userPersonas.GetAllPersonas();
            if (App.algorithm == "sar")
            {
                sarCheck = "True";
                lgbmCheck = "False";
            } 
            else
            {
                sarCheck = "False";
                lgbmCheck = "True";
            }
            
            OpenGitHubLink = new Command(OpenGitHub);
        }

        /// <summary>
        /// Returns corresponding algorithm description 
        /// </summary>
        public string GetDescription(string key)
        {
            return algorithmDescription[key];
        }

        private async void OpenGitHub()
        {
            var builder = new UriBuilder(Resources.GitHubLink);
            await Browser.OpenAsync(builder.Uri, BrowserLaunchMode.SystemPreferred).ConfigureAwait(false);
        }

    }
}
