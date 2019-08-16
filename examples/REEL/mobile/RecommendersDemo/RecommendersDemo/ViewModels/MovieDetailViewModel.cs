using RecommendersDemo.Models;
using System;
using System.Diagnostics.Contracts;
using Xamarin.Forms;

namespace RecommendersDemo.ViewModels
{
    public class MovieDetailViewModel : BaseViewModel
    {

        private readonly UserMoviePreferences preferences;

        //the name of the icons to indicate if a movie is liked or not 
        private const string likedMovieIcon = "after_like.png";
        private const string notLikedMovieIcon = "before_like.png";

        public Movie Movie { get; }
        public Command<Movie> LikeClicked { get; }
        
        //property that is data-bound to the Like button image source 
        private string movieLikeButtonImageSource = notLikedMovieIcon;
        public string MovieLikeButtonImageSource {
            get{ return movieLikeButtonImageSource; }
            set{ SetProperty(ref movieLikeButtonImageSource, value); }
        }

        public MovieDetailViewModel(Movie movie, UserMoviePreferences preferences)
        {
            Contract.Requires(preferences != null);
            this.preferences = preferences;
            Movie = movie;
            MovieLikeButtonImageSource = IsMovieLiked() ? likedMovieIcon : notLikedMovieIcon;
            LikeClicked = new Command<Movie>(ToggleLike);
            preferences.CollectionChanged += UpdateMovieLikeButtonSource;
        }

        /// <summary>
        /// Whenever user movie preferences gets updated, will check itself to see if it needs to update the like button source
        /// </summary>
        /// <param name="sender">The class that triggered the event</param>
        /// <param name="e">More detailed information about what change occurred</param>
        private void UpdateMovieLikeButtonSource(object sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
        {
            MovieLikeButtonImageSource = IsMovieLiked() ? likedMovieIcon : notLikedMovieIcon;
        }

        /// <summary>
        /// Toggles the like/not-liked state of the movie by adding or removing it to UserMoviePreferences and sets the appropriate like-button image 
        /// </summary>
        /// <param name="movie">The movie to be toggled</param>
        public void ToggleLike(Movie movie)
        {
            if (!preferences.RemoveIfPresent(movie))
            {
                preferences.AddPreference(movie);
                MovieLikeButtonImageSource = likedMovieIcon;
            }
            else
            {
                MovieLikeButtonImageSource = notLikedMovieIcon;
            }
        }

        /// <summary>
        /// Returns if Movie is liked by user 
        /// </summary>
        /// <returns> Result true if movie is in UserMoviePreferences, false otherwise </returns>
        private bool IsMovieLiked()
        {
            return preferences.HasMovie(Movie);
        }

        public override bool Equals(object obj)
        {
            var item = obj as MovieDetailViewModel;
            if (item == null)
            {
                return false;
            }

            return this.Movie.ItemID.Equals(item.Movie.ItemID, StringComparison.Ordinal);
        }

        public override int GetHashCode()
        {
            return this.Movie.ItemID.GetHashCode();
        }
    }
}
