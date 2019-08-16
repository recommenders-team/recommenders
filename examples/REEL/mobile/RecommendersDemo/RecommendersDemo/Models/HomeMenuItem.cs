using System;
using System.Collections.Generic;
using System.Text;

namespace RecommendersDemo.Models
{
    public enum MenuItemType
    {
        Browse,
        UserMoviePreferences,
        Search
    }
    public class HomeMenuItem
    {
        public MenuItemType Id { get; set; }

        public string Title { get; set; }
    }
}
