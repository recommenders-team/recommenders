using RecommendersDemo.Models;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using PanCardView.Extensions;
using System.ComponentModel;
using System.Windows.Input;
using Xamarin.Forms;
using System.Linq;
using RecommendersDemo.Services;

namespace RecommendersDemo.ViewModels
{
    /// <summary>
    /// Onboarding process is responsible for setting up a profile for the user   
    /// and asks user to rate certain movies on startup
    /// </summary>
    class OnboardingMovieSelectionViewModel : BaseViewModel
    {
        private UserMoviePreferences preferences;
        private readonly RestClient backend;
        public string PreferredGenres;
        public IList<MovieDetailViewModel> movieList;
        public IList<MovieDetailViewModel> MovieList {
            get { return movieList; }
        }

        /// <summary>
        /// Determines if the auto navigation is running, 
        /// i.e whether cardsindex are changed without user input
        /// </summary>
        public bool IsAutoAnimationRunning { get; set; }

        /// <summary>
        /// Determines if the UserInteration is running, 
        /// i.e. whether user is changing the cards
        /// </summary>
        public bool IsUserInteractionRunning { get; set; }
        
        /// <summary>
        /// Changes swiping interaction
        /// </summary>
        public ICommand PanPositionChangedCommand { get; }

        /// <summary>
        /// Remove current card from carousel
        /// </summary>
        public ICommand RemoveCurrentItemCommand { get; }

        /// <summary>
        /// Go to last card
        /// </summary>
        public ICommand GoToLastCommand { get; }

        public ObservableCollection<object> Items { get; }

        private int _currentIndex;
        private int _ImageCount = 1058;

        public Color nextButtonBackgroundColor;
        public Color NextButtonBackgroundColor
        {
            get { return nextButtonBackgroundColor; }
            set { SetProperty(ref nextButtonBackgroundColor, value); }
        }

        public Color buttonBorder;
        public Color ButtonBorder
        {
            get { return buttonBorder; }
            set { SetProperty(ref buttonBorder, value); }
        }

        public int CurrentIndex
        {
            get => _currentIndex;
            set
            {
                SetProperty(ref _currentIndex, value);
            }
        }

        public OnboardingMovieSelectionViewModel(UserMoviePreferences preferences, RestClient backend)
        {
            nextButtonBackgroundColor = Color.Transparent;
            buttonBorder = (Color)Application.Current.Resources["AccentColor"];
            this.preferences = preferences;
            PreferredGenres = preferences.GetGenres().ToString();
            this.backend = backend;
            movieList = new List<MovieDetailViewModel>();
            Items = new ObservableCollection<object>();
            getMovies();
  

            PanPositionChangedCommand = new Command(v =>
            {
                if (IsAutoAnimationRunning || IsUserInteractionRunning)
                {
                    return;
                }

                var index = CurrentIndex + (bool.Parse(v.ToString()) ? 1 : -1);
                if (index < 0 || index >= Items.Count)
                {
                    return;
                }
                CurrentIndex = index;
            });

            RemoveCurrentItemCommand = new Command(() =>
            {
                if (!Items.Any())
                {
                    return;
                }
                Items.RemoveAt(CurrentIndex.ToCyclingIndex(Items.Count));
            });

            GoToLastCommand = new Command(() =>
            {
                CurrentIndex = Items.Count - 1;
            });
        }

        /// <summary>
        /// Get movies from backend and add them to the carousel for onboarding
        /// </summary>
        private async void getMovies()
        {
            var movieOnboarding = await backend.GetMovieOnboarding(preferences.GetGenres()).ConfigureAwait(true);
            movieList = movieOnboarding.Select(x => new MovieDetailViewModel(x, preferences)).ToList();
            foreach (MovieDetailViewModel movie in movieList)
            {
                Items.Add(new
                {
                    movie = movie, Ind = _ImageCount++,
                });
            }
           
        }

        /// <summary>
        /// Changes button color if user satisfies certain conditions to move onto next onboarding screen
        /// </summary>
        public bool ChangeButtonColor()
        {
            if (preferences.GetAllPreferences().Count > 0)
            {
                NextButtonBackgroundColor = (Color)Application.Current.Resources["AccentColor"];
                ButtonBorder = (Color)Application.Current.Resources["AccentColor"];
                return true;
            }
            else
            {
                NextButtonBackgroundColor = Color.Transparent;
                ButtonBorder = (Color)Application.Current.Resources["AccentColor"];
                return false;
            }
        }

    }
}

