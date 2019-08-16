using Microsoft.VisualStudio.TestTools.UnitTesting;
using RecommendersDemo;
using RecommendersDemo.Models;
using RecommendersDemo.Services;
using System;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Data;

namespace UnitTests
{
    /// <summary>
    /// Test Class to make it possible to test the singleton UserMoviePreferences
    /// </summary>
    public class UserMoviePreferencesTestObject : UserMoviePreferences
    {
        public int CollectionChangedCallCount = 0;
        public UserMoviePreferencesTestObject() : base()
        {
            this.CollectionChanged += new NotifyCollectionChangedEventHandler(UserMoviePreferecesTestObject_CollectionChanged);
        }

        private void UserMoviePreferecesTestObject_CollectionChanged(object sender, System.Collections.Specialized.NotifyCollectionChangedEventArgs e)
        {
            CollectionChangedCallCount++;
        }
    }

    [TestClass]
    public class UserMoviePreferencesTests
    {
        private Movie getSingleMovie()
        {
            Movie movie = new Movie();
            movie.Title = "Avengers: Endgame";
            movie.Year = "2019";
            movie.Genre = new List<string>{ "Action"};
            movie.Overview = "dummySummary";
            movie.ItemID = "456";
            movie.Imageurl = "dummyURL";
            movie.Prediction = "dummyPrediction";
            return movie;
        }

        private List<Movie> getMultipleMovies()
        {
            List<Movie> movies = new List<Movie>();

            Movie movie1 = new Movie();
            movie1.Title = "Avengers: Endgame";
            movie1.Year = "2019";
            movie1.Genre = new List<string> { "Action" };
            movie1.Overview = "dummySummary";
            movie1.ItemID = "456";
            movie1.Imageurl = "dummyURL";
            movie1.Prediction = "dummyPrediction";

            Movie movie2 = new Movie();
            movie2.Title = "Into the Spiderverse";
            movie2.Year = "2018";
            movie2.Genre = new List<string> { "Action" };
            movie2.Overview = "dummySummary";
            movie2.ItemID = "345";
            movie2.Imageurl = "dummyURL";
            movie2.Prediction = "dummyPrediction";

            Movie movie3 = new Movie();
            movie3.Title = "Pride and Prejudice";
            movie3.Year = "2003";
            movie3.Genre = new List<string> { "Romance" };
            movie3.Overview = "dummySummary";
            movie3.ItemID = "783";
            movie3.Imageurl = "dummyURL";
            movie3.Prediction = "dummyPrediction";

            Movie movie4 = new Movie();
            movie4.Title = "Inglorious Bastards";
            movie4.Year = "2002";
            movie4.Genre = new List<string> { "Action|Comedy" };
            movie4.Overview = "";
            movie4.ItemID = "1002";
            movie4.Imageurl = "dummyURL";
            movie4.Prediction = "dummyPrediction";

            movies.Add(movie1);
            movies.Add(movie2);
            movies.Add(movie3);
            movies.Add(movie4);

            return movies;
        }

        private UserMoviePreferences testPreference;

        [TestInitialize]
        public void Setup()
        {
            testPreference = new UserMoviePreferencesTestObject();
        }

        [TestCleanup]
        public void TearDown()
        {
            testPreference.ClearPreferences();
        }

        [TestMethod]
        public void ClearingPreferencesTest()
        {
            testPreference.AddPreference(this.getSingleMovie());
            testPreference.ClearPreferences();
            Assert.AreEqual(0, testPreference.GetAllPreferences().Count);
            Assert.AreEqual(2, ((UserMoviePreferencesTestObject)testPreference).CollectionChangedCallCount);
        }

        [TestMethod]
        public void AddSingleMovieTest()
        {
            testPreference.AddPreference(this.getSingleMovie());
            var movies = testPreference.GetAllPreferences();

            Assert.AreEqual(1, movies.Count);
            Assert.IsTrue(movies.Contains(this.getSingleMovie()));
            Assert.AreEqual(1, ((UserMoviePreferencesTestObject)testPreference).CollectionChangedCallCount);
        }

        /// <summary>
        /// This checks that an error is thrown. This is inconsistent with "AddMultiplePreferences", 
        /// which does not throw an error when it has a duplicate inside (but it doesn't add a movie again, it just doesn't thrown an exception.)
        /// Until we are sure we are using AddMultiplePreferences for something other than loading fake data, we will leave the inconsistences. 
        /// </summary>
        [TestMethod]
        public void AddSingleDuplicateTest()
        {
            testPreference.AddPreference(this.getSingleMovie());
            Action addMovie = () => testPreference.AddPreference(this.getSingleMovie());
            var movies = testPreference.GetAllPreferences();
            Assert.ThrowsException<DuplicateNameException>(addMovie);
        }

        [TestMethod]
        public void AddDuplicateInMultipleTest()
        {
            var movieList = this.getMultipleMovies();
            movieList.Add(this.getSingleMovie());
            testPreference.AddMultiplePreferences(movieList);
            var movies = testPreference.GetAllPreferences();
            Assert.AreEqual(4, movies.Count);
            Assert.IsTrue(movies.Contains(movieList[0]));
            Assert.IsTrue(movies.Contains(movieList[1]));
            Assert.IsTrue(movies.Contains(movieList[2]));
            Assert.IsTrue(movies.Contains(movieList[3]));
            Assert.AreEqual(1, ((UserMoviePreferencesTestObject)testPreference).CollectionChangedCallCount);
        }

        [TestMethod]
        public void AddSingleDuplicateAfterMultipleAdditionsTest()
        {
            var movieList = this.getMultipleMovies();
            testPreference.AddMultiplePreferences(movieList);
            Action addMovie = () => testPreference.AddPreference(this.getSingleMovie());
            Assert.ThrowsException<DuplicateNameException>(addMovie);
        }

        [TestMethod]
        public void AddListWithDuplicateToExistingPreferencesListTest()
        {
            testPreference.AddPreference(this.getSingleMovie());
            testPreference.AddMultiplePreferences(this.getMultipleMovies());
            Assert.AreEqual(4, testPreference.GetAllPreferences().Count);
        }

        [TestMethod]
        public void AddMultipleMoviesTest()
        {
            var movieList = this.getMultipleMovies();
            testPreference.AddMultiplePreferences(movieList);
            var movies = testPreference.GetAllPreferences();
            Assert.AreEqual(4, movies.Count);
            Assert.IsTrue(movies.Contains(movieList[0]));
            Assert.IsTrue(movies.Contains(movieList[1]));
            Assert.IsTrue(movies.Contains(movieList[2]));
            Assert.IsTrue(movies.Contains(movieList[3]));
        }

        [TestMethod]
        public void PreferencesAreSortedTest()
        {
            var movieList = this.getMultipleMovies();
            testPreference.AddMultiplePreferences(movieList);
            var movies = testPreference.GetAllPreferences();
            movieList.Sort();
            Assert.AreEqual(movieList[0], movies[0]);
            Assert.AreEqual(movieList[1], movies[1]);
            Assert.AreEqual(movieList[2], movies[2]);
            Assert.AreEqual(movieList[3], movies[3]);
        }

        [TestMethod]
        public void RemovePreferenceTest()
        {
            var movieList = this.getMultipleMovies();
            testPreference.AddMultiplePreferences(movieList);
            testPreference.RemoveIfPresent(movieList[0]);
            var movies = testPreference.GetAllPreferences();
            Assert.IsFalse(movies.Contains(movieList[0]));
            Assert.AreEqual(3, movies.Count);
            Assert.IsTrue(movies.Contains(movieList[1]));
            Assert.IsTrue(movies.Contains(movieList[2]));
            Assert.IsTrue(movies.Contains(movieList[3]));
            Assert.AreEqual(2, ((UserMoviePreferencesTestObject)testPreference).CollectionChangedCallCount);
        }
    }
}
