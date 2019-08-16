using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using RecommendersDemo.Properties;

namespace RecommendersDemo.Models
{
    public class Movie : IComparable
    {
        [JsonProperty]
        public string ItemID { get; set; }

        [JsonProperty]
        public string Title { get; set; }

        [JsonProperty]
        public List<string> Genre { get; set; }

        [JsonProperty]
        public string Year { get; set; }

        [JsonProperty]
        public string Imageurl { get; set; }

        [JsonProperty]
        public string Overview { get; set; }

        [JsonProperty("prediction")]
        public string Prediction { get; set; }

        public override bool Equals(object obj)
        {
            var item = obj as Movie;
            if (item == null)
            {
                return false;
            }

            return this.ItemID.Equals(item.ItemID, StringComparison.Ordinal);
        }

        public override int GetHashCode()
        {
            return this.ItemID.GetHashCode();
        }

        public int CompareTo(object obj)
        {
            if (obj == null)
            {
                return 1;
            }

            Movie otherMovie = obj as Movie;
            if (otherMovie != null)
            {
                return String.Compare(this.ItemID, otherMovie.ItemID, StringComparison.Ordinal);
            } else
            {
                throw new ArgumentException(Resources.NotMovieObjectErrorMessage);
            }
        }

        public static bool operator ==(Movie first, Movie second)
        {
            if(object.ReferenceEquals(first, null))
            {
                return object.ReferenceEquals(second, null);
            }
            return first.Equals(second);
        }

        public static bool operator !=(Movie first, Movie second)
        {
            if (object.ReferenceEquals(first, null))
            {
                return !object.ReferenceEquals(second, null);
            }
            return !first.Equals(second);
        }

        public static bool operator < (Movie first, Movie second) {
            if (first == null || second == null)
            {
                return false;
            }
            int comparison = first.CompareTo(second);
            if (comparison > -1)
            {
                return false;
            }
            else
            {
                return true;
            }
        }

        public static bool operator > (Movie first, Movie second)
        {
            if (first == null)
            {
                return false;
            }
            int comparison = first.CompareTo(second);
            if (comparison < 1)
            {
                return false;
            }
            else
            {
                return true;
            }
        }

        public static bool operator >= (Movie first, Movie second)
        {
            if (first == null)
            {
                return false;
            }
            int comparison = first.CompareTo(second);
            if (comparison == -1)
            {
                return false;
            }
            else
            {
                return true;
            }
        }

        public static bool operator <= (Movie first, Movie second)
        {
            if (first == null)
            {
                return false;
            }
            int comparison = first.CompareTo(second);
            if (comparison == 1)
            {
                return false;
            }
            else
            {
                return true;
            }
        }
    }
 }
