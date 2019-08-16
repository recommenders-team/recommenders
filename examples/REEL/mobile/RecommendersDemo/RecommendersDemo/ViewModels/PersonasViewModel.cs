using RecommendersDemo.Models;
using RecommendersDemo.Services;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.ComponentModel;
using System.Threading.Tasks;
using Xamarin.Essentials;
using Xamarin.Forms;

namespace RecommendersDemo.ViewModels
{
    class PersonasViewModel : BaseViewModel
    {
        public delegate void ErrorHandler(Exception ex);

        private readonly ErrorHandler errorHandler;
        private readonly UserPersonas userPersonas;

        public List<PersonaWrapper> personas;
        public Persona chosenPersona;

        public ObservableCollection<PersonaContainer> pairedPersonas;
        public ObservableCollection<PersonaContainer> PairedPersonas
        {
            get { return pairedPersonas; }
            set { SetProperty(ref pairedPersonas, value); }
        }

        public Color nextButtonBackgroundColor;
        public Color NextButtonBackgroundColor
        {
            get { return nextButtonBackgroundColor; }
            set { SetProperty(ref nextButtonBackgroundColor, value); }
        }

        public Color buttonBorder;
        public Color ButtonBorder
        {
            get { return buttonBorder; }
            set { SetProperty(ref buttonBorder, value); }
        }

        public PersonasViewModel(ErrorHandler errorHandler)
        {
            this.errorHandler = errorHandler;
            Title = "Personas";
            nextButtonBackgroundColor = Color.Transparent;
            buttonBorder = (Color)Application.Current.Resources["AccentColor"];
            personas = new List<PersonaWrapper>();
            pairedPersonas = new ObservableCollection<PersonaContainer>();
            userPersonas = UserPersonas.GetInstance();
            updatePairedPersonas();
        }

        public void SetChosenPersona(string personaName)
        {
            chosenPersona = personas.Find(x => x.persona.Name.Equals(personaName, StringComparison.Ordinal)).persona;
        }

        /// <summary>
        /// Sets all the checkmarks to not visible for the personas then enables the personas who was clicked on to show the checkmark
        /// </summary>
        /// <param name="persona_name"></param>
        public void ToggleCheckmark(string personaName)
        {
            foreach (PersonaWrapper personaWrapper in personas)
            {
                // hide all the checkmarks           
                if (personaWrapper.persona.Name.Equals(personaName, StringComparison.Ordinal))
                {
                    personaWrapper.ShowCheckmark = "True";
                }
                else
                {
                    personaWrapper.ShowCheckmark = "False";
                }
            }
        }

        /// <summary>
        /// Called after the GetPersonas method, takes the persona objects and puts them into a PersonaContainer object that holds 2 persona
        /// objects then adds this container object to the pairedPersonas list. This is necessary in order to display the personas in a 
        /// 2 column list.
        /// </summary>
        /// <return>None</return>
        private async void updatePairedPersonas()
        {
            try
            {
                pairedPersonas.Clear();
                await userPersonas.PopulatePersonasFromBackend().ConfigureAwait(true);
                personas = userPersonas.GetAllPersonas();

                for (int i = 0; i < personas.Count; i += 2)
                {
                    if (i + 1 < personas.Count)
                    {
                        this.pairedPersonas.Add(new PersonaContainer(personas[i], personas[i + 1]));
                    }
                    else
                    {
                        this.pairedPersonas.Add(new PersonaContainer(personas[i]));
                    }
                }
            }
            catch (BadBackendRequestException e)
            {
                errorHandler(e);
            }
        }

        /// <summary>
        /// Changes button color if user satisfies certain conditions to move onto next onboarding screen
        /// </summary>
        public void SetNextButtonBackgroundColor()
        {
            foreach (PersonaWrapper personaWrapper in personas)
            {
                if (personaWrapper.ShowCheckmark.Equals("True", StringComparison.Ordinal))
                {
                    NextButtonBackgroundColor = (Color)Application.Current.Resources["AccentColor"];
                    ButtonBorder = (Color)Application.Current.Resources["AccentColor"];
                }
            }
        }
    }
}
