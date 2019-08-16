using Microsoft.VisualStudio.TestTools.UnitTesting;
using RecommendersDemo.ViewModels;
using RecommendersDemo.Models;

namespace UnitTests
{
    [TestClass]
    public class MovieDetailViewModelTests
    {
        private MovieDetailViewModel viewModel;
        private Movie movie;
        private UserMoviePreferencesTestObject preferences;
        private readonly string fakeItemID = "123";
        private readonly string fakeURL = "someURL";
        private readonly string fakeOverview = "Some overview or summary here.";
        private readonly string fakePrediction = ".99";
        private readonly string fakeTitle = "A ScaryTitle";
        private readonly string fakeYear = "1998";

        [TestInitialize]
        public void Setup()
        {
            movie = new Movie();
            movie.ItemID = fakeItemID;
            movie.Imageurl = fakeURL;
            movie.Overview = fakeOverview;
            movie.Prediction = fakePrediction;
            movie.Title = fakeTitle;
            movie.Year = fakeYear;

            preferences = new UserMoviePreferencesTestObject ();
        }


        [TestCleanup]
        public void TearDown()
        {
            viewModel = null;
            movie = null;
            preferences.ClearPreferences();
            preferences = null;
        }

        [TestMethod]
        public void MovieArgumentNotMutatedTest()
        {
            viewModel = new MovieDetailViewModel(movie, preferences);
            Assert.AreEqual(fakeItemID, viewModel.Movie.ItemID);
            Assert.AreEqual(fakeURL, viewModel.Movie.Imageurl);
            Assert.AreEqual(fakeOverview, viewModel.Movie.Overview);
            Assert.AreEqual(fakePrediction, viewModel.Movie.Prediction);
            Assert.AreEqual(fakeTitle, viewModel.Movie.Title);
            Assert.AreEqual(fakeYear, viewModel.Movie.Year);
        }

        [TestMethod]
        public void ToggleLikeWhenRemovingMovieFromPreferencesTest()
        {
            viewModel = new MovieDetailViewModel(movie, preferences);
            preferences.AddPreference(movie);
            viewModel.ToggleLike(movie);
            Assert.AreEqual("before_like.png", viewModel.MovieLikeButtonImageSource);
        }

        [TestMethod]
        public void ToggleLikeMovieNotInPreferencesTest()
        {
            viewModel = new MovieDetailViewModel(movie, preferences);
            viewModel.ToggleLike(movie);
            Assert.AreEqual("after_like.png", viewModel.MovieLikeButtonImageSource);
        }

        [TestMethod]
        public void ConstructDetailPageWithMovieAlreadyLikedTest()
        {
            preferences.AddPreference(movie);
            viewModel = new MovieDetailViewModel(movie, preferences);

            Assert.AreEqual("after_like.png", viewModel.MovieLikeButtonImageSource);
        }

        [TestMethod]
        public void ConstructDetailPageWithMovieNotLikedTest()
        {
            viewModel = new MovieDetailViewModel(movie, preferences);
            Assert.AreEqual("before_like.png", viewModel.MovieLikeButtonImageSource);
        }

        [TestMethod]
        public void PreferenceObjectChangesAndViewModelUpdatesTest()
        {
            viewModel = new MovieDetailViewModel(movie, preferences);
            Assert.AreEqual("before_like.png", viewModel.MovieLikeButtonImageSource);
            preferences.AddPreference(movie);
            Assert.AreEqual("after_like.png", viewModel.MovieLikeButtonImageSource);
        }

    }
}
