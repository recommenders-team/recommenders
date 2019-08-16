using RecommendersDemo.Models;
using System.Collections.Generic;

namespace RecommendersDemo.ViewModels
{
    public class PersonaWrapper : BaseViewModel
    {
        public Persona persona { get; set; }

        private string showCheckmark = "False";

        public string ShowCheckmark
        {
            get => showCheckmark;
            set
            {
                SetProperty(ref showCheckmark, value);
            }
        }

        public PersonaWrapper(List<Movie> likedMovies, string name)
        {
            this.persona = new Persona(likedMovies, name);
        }

        public PersonaWrapper(Persona persona)
        {
            this.persona = persona;
        }
    }
}
