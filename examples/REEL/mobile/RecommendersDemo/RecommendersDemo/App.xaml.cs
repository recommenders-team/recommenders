using System;
using Xamarin.Forms;
using Xamarin.Forms.Xaml;
using RecommendersDemo.Services;
using RecommendersDemo.Views;
using Microsoft.AppCenter;
using Microsoft.AppCenter.Analytics;
using Microsoft.AppCenter.Crashes;

namespace RecommendersDemo
{
    public partial class App : Application
    {
        public static string algorithm { get; set; }
        private Page mainTabbedPage;

        public App()
        {
            InitializeComponent();
            MainPage = new LandingPage();
            algorithm = "sar";
        }

        protected override void OnStart()
        {
            // Handle when your app starts
            mainTabbedPage = new MainPage();
        }

        protected override void OnSleep()
        {
            // Handle when your app sleeps
        }

        protected override void OnResume()
        {
            // Handle when your app resumes
        }
    }
}
