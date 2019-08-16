using Microsoft.Extensions.DependencyModel;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using Moq;
using RecommendersDemo.Models;
using RecommendersDemo.Services;
using RecommendersDemo.ViewModels;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;


namespace UnitTests
{
    [TestClass]
    public class BrowseViewModelTests
    {
        private BrowseViewModel viewModel;
        private BrowseViewModel faultyViewModel;
        private Mock<RestClient> mockClient;
        private Mock<RestClient> faultyMockClient;
        private UserMoviePreferencesTestObject preferences;
        private Exception e;
        private bool exceptionHandled;
        private Dictionary<string, List<Movie>> popularMovies = new Dictionary<string, List<Movie>>{
            {Genre.Action.ToString(), new List<Movie>{ new Movie() { ItemID = "123" } } },
            {Genre.Adventure.ToString(), new List<Movie>{ new Movie() { ItemID = "234" } } },
            {Genre.Comedy.ToString(), new List<Movie>{ new Movie() { ItemID = "345" } } },
        };

        [TestInitialize]
        public void Setup()
        {
            exceptionHandled = false;
            preferences = new UserMoviePreferencesTestObject();

            mockClient = new Mock<RestClient>();
            var genreListTask = Task.FromResult<IDictionary<string, List<Movie>>>(popularMovies);
            var recommendedMoviesTask = Task.FromResult<IList<Movie>>(new List<Movie> { new Movie() { ItemID = "111" } });
            mockClient.Setup(x => x.GetPopularMovies(It.IsAny<List<Genre>>())).Returns(genreListTask);
            mockClient.Setup(x => x.GetMovieRecommendationsFromPreferences(It.IsAny<UserMoviePreferences>())).Returns(recommendedMoviesTask);

            faultyMockClient = new Mock<RestClient>();
            viewModel = new BrowseViewModel(ex => { e = ex; exceptionHandled = true; }, preferences, mockClient.Object);
        }

        [TestMethod]
        public void ActionMovieInitializedCorrectlyTest()
        {
            MovieDetailViewModel comparisonActionMovie = new MovieDetailViewModel(new Movie { ItemID = "123" }, preferences);
            Assert.AreEqual(1, viewModel.PopularActionMovies.Count);
            Assert.AreEqual(comparisonActionMovie, viewModel.PopularActionMovies[0]);
        }

        [TestMethod]
        public void ComedyMoviesInitializedCorrectlyTest()
        {
            MovieDetailViewModel comparisonComedyMovie = new MovieDetailViewModel(new Movie() { ItemID = "345" }, preferences);
            Assert.AreEqual(1, viewModel.PopularComedyMovies.Count);
            Assert.AreEqual(comparisonComedyMovie, viewModel.PopularComedyMovies[0]);
        }

        [TestMethod]
        public void RecommendedMoviesUpdatedCorrectlyTest()
        {
            MovieDetailViewModel comparisonMovie = new MovieDetailViewModel(new Movie() { ItemID = "111" }, preferences);
            var task = viewModel.UpdateMovieRecommendations();
            Assert.AreEqual(1, viewModel.RecommendedMovies.Count);
            Assert.AreEqual(comparisonMovie, viewModel.RecommendedMovies[0]);
        }

        [TestMethod]
        public void PullRefreshUpdatesRecommendedMoviesTest()
        {
            var task = viewModel.RefreshRequestRecommendations();
            mockClient.Verify(mock => mock.GetPopularMovies(It.IsAny<List<Genre>>()), Times.Once());
        }

        [TestMethod]
        public void BackendErrorWhenRequestingPopularMoviesTest()
        {
            try
            {
                faultyMockClient.Setup(x => x.GetPopularMovies(It.IsAny<List<Genre>>())).Throws(new BadBackendRequestException());
                faultyViewModel = new BrowseViewModel(ex => { e = ex; exceptionHandled = true; }, preferences, faultyMockClient.Object);
            }
            catch
            {
                Assert.Fail();
            }
            Assert.IsTrue(exceptionHandled);
        }

        [TestMethod]

        public void BackendErrorWhenRequestingRecommendedMoviesTest()
        {
            try
            {
                var genreListTask = Task.FromResult<IDictionary<string, List<Movie>>>(popularMovies);
                faultyMockClient.Setup(x => x.GetPopularMovies(It.IsAny<List<Genre>>())).Returns(genreListTask);
                faultyMockClient.Setup(x => x.GetMovieRecommendationsFromPreferences(It.IsAny<UserMoviePreferences>())).Throws(new BadBackendRequestException());
                faultyViewModel = new BrowseViewModel(ex => { e = ex; exceptionHandled = true; }, preferences, faultyMockClient.Object);
                var task = faultyViewModel.UpdateMovieRecommendations();
            }
            catch
            {
                Assert.Fail();
            }
            Assert.IsTrue(exceptionHandled);
        }

        [TestMethod]
        public void EmptyPopularMoviesListReturnedTest()
        {
            var genreListTask = Task.FromResult<IDictionary<string, List<Movie>>>(new Dictionary<string, List<Movie>>());
            faultyMockClient.Setup(x => x.GetPopularMovies(It.IsAny<List<Genre>>())).Returns(genreListTask);

            try
            {
                faultyViewModel = new BrowseViewModel(ex => { e = ex; exceptionHandled = true; }, preferences, faultyMockClient.Object);
            }
            catch
            {
                Assert.Fail();
            }
            Assert.IsFalse(exceptionHandled);
            Assert.AreEqual(0, faultyViewModel.PopularActionMovies.Count);
            Assert.AreEqual(0, faultyViewModel.PopularComedyMovies.Count);
        }

        [TestMethod]
        public void EmptyRecommendedMoviesListReturnedTest()
        {
            var genreListTask = Task.FromResult<IDictionary<string, List<Movie>>>(popularMovies);
            var recommendedMoviesTask = Task.FromResult<IList<Movie>>(new List<Movie>());
            faultyMockClient.Setup(x => x.GetPopularMovies(It.IsAny<List<Genre>>())).Returns(genreListTask);
            faultyMockClient.Setup(x => x.GetMovieRecommendationsFromPreferences(It.IsAny<UserMoviePreferences>())).Returns(recommendedMoviesTask);

            try
            {
                faultyViewModel = new BrowseViewModel(ex => { e = ex; exceptionHandled = true; }, preferences, faultyMockClient.Object);
                var task = faultyViewModel.UpdateMovieRecommendations();
            }
            catch
            {
                Assert.Fail();
            }

            Assert.IsFalse(exceptionHandled);
            Assert.AreEqual(0, faultyViewModel.RecommendedMovies.Count);
        }
    }
}
