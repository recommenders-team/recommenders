using Xamarin.Forms;
using RecommendersDemo.Models;

namespace RecommendersDemo.ViewModels
{

    public class SelectGenresViewModel : BaseViewModel
    {
        private UserMoviePreferences preferences;
        public Command<string> ClickGenre { get; }

        private string actionVisibility = "False";
        private string animatedVisibility = "False";
        private string comedyVisibility = "False";
        private string dramaVisibility = "False" ;
        private string horrorVisibility = "False" ;
        private string romanceVisibility = "False" ;

        public SelectGenresViewModel()
        { 
            ClickGenre = new Command<string>(ToggleGenre);
            preferences = UserMoviePreferences.getInstance();
            nextButtonBackgroundColor = Color.Transparent;
            buttonBorder = (Color)Application.Current.Resources["AccentColor"];
        }

        private Color nextButtonBackgroundColor;
        public Color NextButtonBackgroundColor
        {
            get { return nextButtonBackgroundColor; }
            set { SetProperty(ref nextButtonBackgroundColor, value); }
        }

        private Color buttonBorder;
        public Color ButtonBorder
        {
            get { return buttonBorder; }
            set { SetProperty(ref buttonBorder, value); }
        }

        void ToggleGenre(string genre)
        {
            if (!preferences.RemoveGenreIfPresent(genre))
            {
                preferences.AddGenre(genre);
                this.SetGenre(genre, true);
            } 
            else
            {
                this.SetGenre(genre, false);
            }
            ChangeButtonColor();
        }

        private void SetGenre(string genre, bool liked)
        {
            switch (genre)
            {
                case "Action":
                    if (liked) { ActionVisibility = "True"; } else { ActionVisibility = "False"; }
                    break;
                case "Animation":
                    if (liked) { AnimatedVisibility = "True"; } else { AnimatedVisibility = "False"; }
                    break;
                case "Comedy":
                    if (liked) { ComedyVisibility = "True"; } else { ComedyVisibility = "False"; }
                    break;
                case "Drama":
                    if (liked) { DramaVisibility = "True"; } else { DramaVisibility = "False"; }
                    break;
                case "Horror":
                    if (liked) { HorrorVisibility = "True"; } else { HorrorVisibility = "False"; }
                    break;
                case "Romance":
                    if (liked) { RomanceVisibility = "True"; } else { RomanceVisibility = "False"; }
                    break;
            }
        }

        public string ActionVisibility
        {
            get => actionVisibility;
            set
            {
                SetProperty(ref actionVisibility, value);
            }
        }

        public string AnimatedVisibility
        {
            get => animatedVisibility;
            set
            {
                SetProperty(ref animatedVisibility, value);
            }
        }

        public string ComedyVisibility
        {
            get => comedyVisibility;
            set
            {
                SetProperty(ref comedyVisibility, value);
            }
        }

        public string DramaVisibility
        {
            get => dramaVisibility;
            set
            {
                SetProperty(ref dramaVisibility, value);
            }
        }

        public string HorrorVisibility
        {
            get => horrorVisibility;
            set
            {
                SetProperty(ref horrorVisibility, value);
            }
        }

        public string RomanceVisibility
        {
            get => romanceVisibility;
            set
            {
                SetProperty(ref romanceVisibility, value);
            }
        }  

        public bool ChangeButtonColor()
        {       
            if (preferences.GetGenres().Count == 2 || preferences.GetGenres().Count == 3)
            {
                NextButtonBackgroundColor = (Color)Application.Current.Resources["AccentColor"];
                ButtonBorder = (Color)Application.Current.Resources["AccentColor"];
                return true;
            }
            else
            {
                NextButtonBackgroundColor = Color.Transparent;
                ButtonBorder = (Color)Application.Current.Resources["AccentColor"];
                return false;
            }       
        }
    }
}
