using RecommendersDemo.ViewModels;
using System;
using System.Collections;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics.Contracts;
using System.Runtime.CompilerServices;

namespace RecommendersDemo.Models
{
    public class Persona
    {
        private List<Movie> likedMovies;

        public string Name { get; set; }

        public string IconSource { get; set; }
        
        public Persona(List<Movie> likedMovies, string name)
        {
            Contract.Requires(name != null);
            this.likedMovies = likedMovies;
            this.Name = name;
            if(name.Equals("Rom Com Tom", StringComparison.Ordinal)) { IconSource = "Tom.png"; }
            if(name.Equals("Cartoon Carly", StringComparison.Ordinal))    { IconSource = "Carly.png"; }
            if(name.Equals("Action Jackson", StringComparison.Ordinal))   { IconSource = "Jackson.png"; }
            if(name.Equals("Joking Jane", StringComparison.Ordinal))      { IconSource = "Jane.png"; }
        }

        public IList<Movie> getLikedMovies()
        {
            return new List<Movie>(likedMovies);
        }
    }
}
