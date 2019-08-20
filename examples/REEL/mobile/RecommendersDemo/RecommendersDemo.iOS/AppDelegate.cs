using System;
using System.Collections.Generic;
using System.Linq;
using FFImageLoading.Forms.Platform;
using Foundation;
using UIKit;
using Xamarin.Forms;
using Lottie.Forms.iOS.Renderers;
using PanCardView.iOS;

namespace RecommendersDemo.iOS
{
    // The UIApplicationDelegate for the application. This class is responsible for launching the 
    // User Interface of the application, as well as listening (and optionally responding) to 
    // application events from iOS.
    [Register("AppDelegate")]
    public partial class AppDelegate : global::Xamarin.Forms.Platform.iOS.FormsApplicationDelegate
    {
        //
        // This method is invoked when the application has loaded and is ready to run. In this 
        // method you should instantiate the window, load the UI into it and then make the window
        // visible.
        //
        // You have 17 seconds to return from this method, or iOS will terminate your application.
        //
        public override bool FinishedLaunching(UIApplication app, NSDictionary options)
        {

            CachedImageRenderer.Init();
            Forms.SetFlags("CollectionView_Experimental");
            global::Xamarin.Forms.Forms.Init();
            CachedImageRenderer.InitImageSourceHandler();
            UINavigationBar.Appearance.SetBackgroundImage(UIImage.FromFile("nav_background.png"), UIBarMetrics.Default);
            UITabBar.Appearance.BackgroundImage = new UIImage();
            UITabBar.Appearance.BackgroundColor = UIColor.FromRGB(39, 59, 81);     
            AnimationViewRenderer.Init();
            CardsViewRenderer.Preserve();
            LoadApplication(new App());

            return base.FinishedLaunching(app, options);
        }

        // <summary>
        // Locks iOS phone orientation to portrait mode.
        // </summary
        public override UIInterfaceOrientationMask GetSupportedInterfaceOrientations(UIApplication application, UIWindow forWindow)
        {     
            return UIInterfaceOrientationMask.Portrait;
        }
    }
}
