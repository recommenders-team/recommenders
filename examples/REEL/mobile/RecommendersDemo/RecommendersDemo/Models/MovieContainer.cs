using System;
using System.Collections.Generic;
using System.Text;
using Newtonsoft.Json;
using RecommendersDemo.ViewModels;

namespace RecommendersDemo.Models
{
    public class MovieContainer
    {
        public MovieDetailViewModel FirstMovie { get; set; }
        public Boolean DisplayFirstMovie { get; set; }
        public MovieDetailViewModel SecondMovie { get; set; }
        public Boolean DisplaySecondMovie { get; set; }


        /**
         * Constructor of the SearchedMovieContainer which holds the first and second movie attributes for 1 row of the search results. These attributes will be binded to labels inside SearchPage.xaml to present search results
         * 
         */
        public MovieContainer(MovieDetailViewModel FirstMovie, MovieDetailViewModel SecondMovie)
        {
            this.FirstMovie = FirstMovie;
            this.DisplayFirstMovie = true;

            this.SecondMovie = SecondMovie;
            this.DisplaySecondMovie = true;
        }

        /**
         * Constructor of the SearchedMovieContainer that holds only the first movie in a row for the search results and disables the presentation of a second movie, typically because there is an odd number of movies returned.
         * Because the search results are presented into 2 columns, one row will only have one movie in it and this constructor holds the information for that special case.
         */
        public MovieContainer(MovieDetailViewModel FirstMovie)
        {
            this.FirstMovie = FirstMovie;
            this.DisplayFirstMovie = true;

            this.DisplaySecondMovie = false;
        }
    }
}
