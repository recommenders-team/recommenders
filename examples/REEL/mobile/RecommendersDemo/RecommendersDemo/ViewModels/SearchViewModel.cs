using RecommendersDemo.Models;
using RecommendersDemo.Services;
using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics.Contracts;
using System.Linq;
using Xamarin.Forms;

namespace RecommendersDemo.ViewModels
{
    public class SearchViewModel : BaseViewModel
    {

        public delegate void ErrorHandler(Exception ex);

        private readonly ErrorHandler errorHandler;
        private RestClient backend;
        private UserMoviePreferences preferences;
        private ObservableCollection<MovieContainer> pairedMovieList = new ObservableCollection<MovieContainer>();
        private IList<Movie> searchedMovies = new List<Movie>(); // searchedMovies
        private bool noResultsMessage;
        public bool NoResultsMessage { get { return noResultsMessage; } set { SetProperty(ref noResultsMessage, value); } }

        public ObservableCollection<MovieContainer> PairedMovieList
        {
            get { return pairedMovieList; }
            set { SetProperty(ref pairedMovieList, value); }
        }

        public SearchViewModel(ErrorHandler errorHandler, RestClient backend)
        {
            this.errorHandler = errorHandler;
            Title = "Search Movies";
            this.backend = backend;
            preferences = UserMoviePreferences.getInstance();
        }

        public async void GetSearchResults(string userInput, string pageNumber)
        {
            Contract.Requires(userInput != null);
            Contract.Requires(pageNumber != null);
            try
            {
                searchedMovies = await backend.getSearchResults(userInput, pageNumber).ConfigureAwait(true);
                pairedMovieList.Clear();
                searchedMovies.ToList();
                if (searchedMovies.Count < 1 || userInput.Length == 0)
                {
                    NoResultsMessage = true;
                } else
                {
                    NoResultsMessage = false;

                    for (int i = 0; i < searchedMovies.Count; i++)
                    {
                        if (searchedMovies[i].Imageurl == null)
                        {
                            searchedMovies[i].Imageurl = "default_image.png";
                        }
                    }

                    for (int i = 0; i < searchedMovies.Count; i += 2) {

                        MovieContainer container = null;
                    
                        if (i + 1 < searchedMovies.Count())
                        {
                            container = new MovieContainer(new MovieDetailViewModel(searchedMovies[i], preferences), new MovieDetailViewModel(searchedMovies[i + 1], preferences));
                        } else
                        {
                            container = new MovieContainer(new MovieDetailViewModel(searchedMovies[i], preferences));
                        }
                        pairedMovieList.Add(container);
                    }                 
                }


            }
            catch (BadBackendRequestException ex)
            {
                errorHandler(ex);
            }
        }
    }
}
