using RecommendersDemo.Models;
using RecommendersDemo.Properties;
using RecommendersDemo.Services;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Threading.Tasks;
using Xamarin.Forms;

namespace RecommendersDemo.ViewModels
{
    public class BrowseViewModel : BaseViewModel
    {

        public delegate void ErrorHandler(Exception ex);

        public Command OnPullDownRefresh { get;  }

        private readonly ErrorHandler errorHandler;
        private readonly RestClient backend;
        private readonly UserMoviePreferences preferences;

        private IReadOnlyList<MovieDetailViewModel> popularComedyMovies;
        public IReadOnlyList<MovieDetailViewModel> PopularComedyMovies
        {
            get { return popularComedyMovies; }
            set { SetProperty(ref popularComedyMovies, value); }
        }

        private IReadOnlyList<MovieDetailViewModel> popularActionMovies;
        public IReadOnlyList<MovieDetailViewModel> PopularActionMovies
        {
            get { return popularActionMovies; }
            set { SetProperty(ref popularActionMovies, value); }
        }

        private IReadOnlyList<MovieDetailViewModel> popularAnimationMovies;
        public IReadOnlyList<MovieDetailViewModel> PopularAnimationMovies
        {
            get { return popularAnimationMovies; }
            set { SetProperty(ref popularAnimationMovies, value); }
        }

        private IReadOnlyList<MovieDetailViewModel> popularDramaMovies;
        public IReadOnlyList<MovieDetailViewModel> PopularDramaMovies
        {
            get { return popularDramaMovies; }
            set { SetProperty(ref popularDramaMovies, value); }
        }

        private IReadOnlyList<MovieDetailViewModel> popularHorrorMovies;
        public IReadOnlyList<MovieDetailViewModel> PopularHorrorMovies
        {
            get { return popularHorrorMovies; }
            set { SetProperty(ref popularHorrorMovies, value); }
        }

        private IReadOnlyList<MovieDetailViewModel> popularRomanceMovies;
        public IReadOnlyList<MovieDetailViewModel>PopularRomanceMovies
        {
            get { return popularRomanceMovies; }
            set { SetProperty(ref popularRomanceMovies, value); }
        }

        private IReadOnlyList<MovieDetailViewModel> recommendedMovies;
        public IReadOnlyList<MovieDetailViewModel> RecommendedMovies
        {
            get { return recommendedMovies; }
            set { SetProperty(ref recommendedMovies, value); }
        }

        private bool isRefreshing = false;
        public bool IsRefreshing
        {
            get { return isRefreshing; }
            set { SetProperty(ref isRefreshing, value); }
        }

        private bool showRecommendationsLabel = false;
        public bool ShowRecommendationsLabel
        {
            get { return showRecommendationsLabel; }
            set { SetProperty(ref showRecommendationsLabel, value); }
        }

        public BrowseViewModel(ErrorHandler errorHandler, UserMoviePreferences preferences, RestClient backend)
        {
            Title = "Browse Movies";
            this.errorHandler = errorHandler;
            this.backend = backend;
            this.preferences = preferences;
            popularActionMovies = new List<MovieDetailViewModel>();
            popularComedyMovies = new List<MovieDetailViewModel>();
            popularAnimationMovies = new List<MovieDetailViewModel>();
            RecommendedMovies = new ObservableCollection<MovieDetailViewModel>();
            this.UpdatePopularMovies();
            OnPullDownRefresh = new Command(async () => await RefreshRequestRecommendations().ConfigureAwait(true));
        }

        /// <summary>
        /// Gets list of popular movies from backend and populates Lists with the movies 
        /// </summary>
        public async void UpdatePopularMovies()
        {
            try
            {
                IDictionary<string, List<Movie>> movies = await backend.GetPopularMovies(new List<Genre>() { Genre.Action, Genre.Comedy, Genre.Animation, Genre.Drama, Genre.Horror, Genre.Romance}).ConfigureAwait(false);
                PopularActionMovies = movies[Genre.Action.ToString()].Select(x => new MovieDetailViewModel(x, preferences)).ToList();
                PopularComedyMovies = movies[Genre.Comedy.ToString()].Select(x => new MovieDetailViewModel(x, preferences)).ToList();
                PopularAnimationMovies = movies[Genre.Animation.ToString()].Select(x => new MovieDetailViewModel(x, preferences)).ToList();
                PopularDramaMovies = movies[Genre.Drama.ToString()].Select(x => new MovieDetailViewModel(x, preferences)).ToList();
                PopularHorrorMovies = movies[Genre.Horror.ToString()].Select(x => new MovieDetailViewModel(x, preferences)).ToList();
                PopularRomanceMovies = movies[Genre.Romance.ToString()].Select(x => new MovieDetailViewModel(x, preferences)).ToList();
            }
            catch (BadBackendRequestException ex)
            {
                errorHandler(ex);
            }
        }

        /// <summary>
        /// Calls backend with a request for recommendations given current liked movies
        /// </summary>
        public async Task UpdateMovieRecommendations()
        {
            try
            {
                IList<Movie> recommendations = await backend.GetMovieRecommendationsFromPreferences(preferences).ConfigureAwait(true);
                RecommendedMovies = recommendations.Select(x => new MovieDetailViewModel(x, preferences)).ToList();
            }
            catch (BadBackendRequestException ex)
            {
                errorHandler(ex);
            }
            return;
        }

        /// <summary>
        /// Called when pull-to-refresh/button is triggered; checks conditions of the request before proceeding
        /// </summary>
        public async Task RefreshRequestRecommendations()
        {
            if (IsRefreshing)
                return;

            if (preferences.NumberOfLikedMovies() < 1)
            {
                errorHandler(new BadBackendRequestException(Resources.LikeMoviesBeforeRecsWarningMessage));
                return;
            }

            try
            {
                IsRefreshing = true;
                await UpdateMovieRecommendations().ConfigureAwait(true);
                ShowRecommendationsLabel = true;
            }
            finally
            {
               IsRefreshing = false;
            }
        }
    }
}