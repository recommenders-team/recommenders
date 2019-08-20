using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Collections.Specialized;
using System.Data;
using RecommendersDemo.Services;

namespace RecommendersDemo.Models
{
    public class UserMoviePreferences: INotifyCollectionChanged
    {
        private SortedSet<Movie> movies;
        private SortedSet<string> genres;
        private SortedSet<string> filters;
        private static UserMoviePreferences obj = new UserMoviePreferences();

        public event NotifyCollectionChangedEventHandler CollectionChanged;

        protected UserMoviePreferences()
        {
            movies = new SortedSet<Movie>();
            genres = new SortedSet<string>();
            filters = new SortedSet<string>();
            filters.Add("Action");
            filters.Add("Adventure");
            filters.Add("Comedy");
            filters.Add("Drama");
            filters.Add("Horror");
            filters.Add("Romance");
            filters.Add("Other");
        }

        public static UserMoviePreferences getInstance()
        {
            return obj;
        }

        /// <summary>
        /// Add a movie to a user's preferences so that it can be used in determining recommendations. 
        /// </summary>
        /// <param name="movie">A movie object that will be added to the User's stored preferences.</param>
        public void AddPreference(Movie movie)
        {
            if(movies.Contains(movie) == true)
            {
                throw new DuplicateNameException("This movie already exists.");
            }
            else
            {
                var eventArgs = new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Add, movie);
                movies.Add(movie);
                CollectionChanged.Invoke(this, eventArgs);
            }
        }

        /// <summary>
        /// Add a list of liked movies to a user's preferred/liked movies
        /// </summary>
        /// <param name="movies">A list of movies</param>
        public void AddMultiplePreferences(List<Movie> movieList)
        {
            var eventArgs = new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Add, movieList);
            movies.UnionWith(movieList);
            CollectionChanged.Invoke(this, eventArgs);
        }

        /// <summary>
        /// Removes a movie from a user's preference if it exists, returns true if successful, false otherwise 
        /// </summary>
        /// <param name="movie">Movie that user would like to have removed from their preferences.</param>
        public bool RemoveIfPresent(Movie movie)
        {
            var eventArgs = new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Remove, movie);
            if (movies.Remove(movie))
            {
                CollectionChanged.Invoke(this, eventArgs);
                return true;
            }
            return false;
        }

        /// <summary>
        /// Get a list of all the movies that a user prefers. 
        /// </summary>
        /// <returns>A list of a user's preferred movies.</returns>
        public IList<Movie> GetAllPreferences()
        {
            return new ReadOnlyCollection<Movie>(new List<Movie>(movies));
        }

        /// <summary>
        /// Returns the user's liked movies as a list of just the movie ids 
        /// </summary>
        /// <returns>The list of ids of the liked movies</returns>
        public List<string> GetListOfMovieIDs()
        {
            List<string> movieIDs = new List<string>();
            foreach(Movie movie in movies)
            {
                movieIDs.Add(movie.ItemID);
            }
            return movieIDs;
        }

        /// <summary>
        /// Clears the user's movie preferences
        /// </summary>
        public void ClearPreferences()
        {
            this.movies = new SortedSet<Movie>();
            var eventArgs = new NotifyCollectionChangedEventArgs(NotifyCollectionChangedAction.Reset);
            CollectionChanged.Invoke(this, eventArgs);
        }

        /// <summary>
        /// Returns the number of movies in the sorted set Movies
        /// </summary>
        /// <returns></returns>
        public int NumberOfLikedMovies()
        {
            return movies.Count;
        }

        /// <summary>
        /// Checks to see if a movie has been liked by the user
        /// </summary>
        /// <param name="movie">The movie being checked</param>
        /// <returns>Result true if the movie has been liked, false otherwise</returns>
        public bool HasMovie(Movie movie)
        {
            return movies.Contains(movie);
        }

        public void AddGenre(string genre)
        {
            genres.Add(genre);
        }

        public void RemoveGenre(string genre)
        {
            genres.Remove(genre);
        }

        public void AddFilter(string genre)
        {
            filters.Add(genre);
        }

        public void RemoveFilter(string genre)
        {
            filters.Remove(genre);
        }

        public IList<string> GetGenres()
        {
            return new List<string>(genres);
        }

        public IList<string> GetFilters()
        {
            return new List<string>(filters);
        }

        public bool RemoveGenreIfPresent(string genre)
        {
            if (genres.Contains(genre))
            {
                genres.Remove(genre);
                return true;
            }
            else
            {
                return false;
            }
        }
    }
}
