using RecommendersDemo.Services;
using RecommendersDemo.ViewModels;
using System.Diagnostics.Contracts;

namespace RecommendersDemo.Models
{
    public class PersonaContainer
    {
        public Persona FirstPersona { get; set; }
        public Persona SecondPersona { get; set; }

        public PersonaWrapper FirstPersonaWrapper { get; set; }

        public PersonaWrapper SecondPersonaWrapper { get; set; }

        public PersonaContainer(PersonaWrapper FirstPersona, PersonaWrapper SecondPersona)
        {
            Contract.Requires(FirstPersona != null && SecondPersona != null);
            this.FirstPersonaWrapper = FirstPersona;
            this.SecondPersonaWrapper = SecondPersona;
            this.FirstPersona = FirstPersona.persona;
            this.SecondPersona = SecondPersona.persona;
        }

        public PersonaContainer(PersonaWrapper FirstPersona)
        {
            Contract.Requires(FirstPersona != null);
            this.FirstPersonaWrapper = FirstPersona;
            this.FirstPersona = FirstPersona.persona;
        }
    }
}
