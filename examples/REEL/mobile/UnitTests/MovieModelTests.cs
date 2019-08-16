using Microsoft.VisualStudio.TestTools.UnitTesting;
using RecommendersDemo.Models;
using System.Collections.Generic;

namespace UnitTests
{
    [TestClass]
    public class MovieModelTests
    {

        [TestMethod]
        public void SelfEqualityTest()
        {
            Movie movie1 = new Movie();
            Assert.AreEqual(movie1, movie1);
        }

        [TestMethod]
        public void TwoEqualItemsEqualityTest()
        {
            Movie movie1 = new Movie();
            movie1.ItemID = "123";
            Movie movie2 = new Movie();
            movie2.ItemID = "123";
            Assert.AreEqual(movie1, movie2);
        }

        [TestMethod]
        public void SimilarNotEqualTest()
        {
            Movie movie1 = new Movie();
            movie1.ItemID = "123";
            movie1.Title = "Toy Story";
            movie1.Genre = new List<string> { "Kids" };
            movie1.Year = "1997";
            movie1.Imageurl = "dummyURL";
            movie1.Overview = "dummyOverview";
            movie1.Prediction = ".99";

            Movie movie2 = new Movie();
            movie2.ItemID = "124";
            movie2.Title = "Toy Story";
            movie2.Genre = new List<string> { "Kids" };
            movie2.Year = "1997";
            movie2.Imageurl = "dummyURL";
            movie2.Overview = "dummyOverview";
            movie2.Prediction = ".99";

            Assert.AreNotEqual(movie1, movie2);
        }

        [TestMethod]
        public void TwoEqualItemsHashTest()
        {
            Movie movie1 = new Movie();
            movie1.ItemID = "123";
            Movie movie2 = new Movie();
            movie2.ItemID = "123";

            Assert.AreEqual(movie1.GetHashCode(), movie2.GetHashCode());
        }

        [TestMethod]
        public void CompareTwoEqualTest()
        {
            Movie movie1 = new Movie();
            movie1.ItemID = "123";
            Movie movie2 = new Movie();
            movie2.ItemID = "123";

            int comparison = movie1.CompareTo(movie2);
            Assert.AreEqual(0, comparison);
        }

        [TestMethod]
        public void CompareTwoDifferentMoviesTest()
        {
            Movie movie1 = new Movie();
            movie1.ItemID = "123";
            Movie movie2 = new Movie();
            movie2.ItemID = "124";

            int comparison1 = movie1.CompareTo(movie2);
            Assert.AreEqual(-1, comparison1);

            int comparison2 = movie2.CompareTo(movie1);
            Assert.AreEqual(1, comparison2);
        }

        [TestMethod]
        public void DifferentPropertiesSameIDTest()
        {
            Movie movie1 = new Movie();
            movie1.ItemID = "123";
            movie1.Title = "Batman Begins (2004)";
            movie1.Genre = new List<string> { "Action" };
            movie1.Imageurl = "dummyURL1";
            movie1.Prediction = ".99";
            movie1.Year = "2004";
            movie1.Overview = "dummyoverview1";
            Movie movie2 = new Movie();
            movie2.ItemID = "123";
            movie2.Title = "Batman Begins";
            movie2.Genre = new List<string> { "Drama" };
            movie2.Imageurl = "dummyURL2";
            movie2.Prediction = ".55";
            movie2.Overview = "dummyoverview2";
            movie2.Year = "2005";

            int comparison = movie1.CompareTo(movie2);
            Assert.AreEqual(0, comparison);
        }

    }
}
