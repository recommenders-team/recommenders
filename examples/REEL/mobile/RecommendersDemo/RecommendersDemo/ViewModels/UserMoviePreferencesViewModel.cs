using RecommendersDemo.Models;
using RecommendersDemo.Services;
using System;
using System.Linq;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics.Contracts;

namespace RecommendersDemo.ViewModels
{
    public class UserMoviePreferencesViewModel : BaseViewModel
    {
        public delegate void ErrorHandler(Exception ex);
        private UserMoviePreferences preferences;

        private readonly ErrorHandler errorHandler;

        private ObservableCollection<Movie> likedMovies;
        public ObservableCollection<Movie> LikedMovies
        {
            get { return likedMovies; }
            set { SetProperty(ref likedMovies, value); }
        }

        private ObservableCollection<MovieContainer> pairedMovieList = new ObservableCollection<MovieContainer>();
        public ObservableCollection<MovieContainer> PairedMovieList
        {
            get { return pairedMovieList; }
            set { SetProperty(ref pairedMovieList, value); }
        }

        public UserMoviePreferencesViewModel(ErrorHandler errorHandler, UserMoviePreferences preferences)
        {
            Contract.Requires(preferences != null);
            Title = "Liked Movies";
            this.errorHandler = errorHandler;
            this.preferences = preferences;
            likedMovies = new ObservableCollection<Movie>(preferences.GetAllPreferences());
            preferences.CollectionChanged += Preferences_CollectionChanged;
        }

        private void Preferences_CollectionChanged(object sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
        {
            LikedMovies = new ObservableCollection<Movie>(preferences.GetAllPreferences());
            UpdatePairedList();
        }

        public void RemoveFromPreference(MovieDetailViewModel movie)
        {
            Contract.Requires(movie != null);
            preferences.RemoveIfPresent(movie.Movie);
        }


        /**
         * Puts all the movies the user has liked, which is stored in the preferences model object, into the local LikedMovies list, then is fit into the movie container
         */
        public void Sync()
        {
            LikedMovies = new ObservableCollection<Movie>(preferences.GetAllPreferences());
            UpdatePairedList();
        }

        public void UpdatePairedList()
        {
            try
            {
                IList<Movie> preferredMovies = GetPreferencesInGenres(preferences.GetFilters());
                pairedMovieList.Clear();
                for (int i = 0; i < preferredMovies.Count; i += 2)
                {
                    MovieContainer container = null;

                    if (i + 1 < preferredMovies.Count)
                    {
                        container = new MovieContainer(new MovieDetailViewModel(preferredMovies[i], preferences), new MovieDetailViewModel(preferredMovies[i + 1], preferences));
                    }
                    else
                    {
                        container = new MovieContainer(new MovieDetailViewModel(preferredMovies[i], preferences));
                    }
                    pairedMovieList.Add(container);
                }
            }
            catch (BadBackendRequestException ex)
            {
                errorHandler(ex);
            }
        }

        public IList<Movie> GetPreferencesInGenres(IList<String> genres)
        {
            List<Movie> movies = new List<Movie>();

            
            var allowedGenres = new[]{ genres };
            List<Movie> result = (from movie in likedMovies
                                  where genres.Intersect(movie.Genre).Any()
                                  select movie).Distinct().ToList();
            return result;
        }

        public void UpdatePairedListForGenre(IList<String> genres)
        {
            IList<string> filteredMovies = preferences.GetFilters();
            ObservableCollection<Movie> likedMoviesInGenre = new ObservableCollection<Movie>(GetPreferencesInGenres(filteredMovies));

            try
            {
                pairedMovieList.Clear();
                for (int i = 0; i < likedMoviesInGenre.Count; i += 2)
                {
                    MovieContainer container = null;

                    if (i + 1 < likedMoviesInGenre.Count)
                    {
                        container = new MovieContainer(new MovieDetailViewModel(likedMoviesInGenre[i], preferences), new MovieDetailViewModel(likedMoviesInGenre[i + 1], preferences));
                    }
                    else
                    {
                        container = new MovieContainer(new MovieDetailViewModel(likedMoviesInGenre[i], preferences));
                    }
                    pairedMovieList.Add(container);
                }
            }
            catch (BadBackendRequestException ex)
            {
                errorHandler(ex);
            }
        }
    }
}
