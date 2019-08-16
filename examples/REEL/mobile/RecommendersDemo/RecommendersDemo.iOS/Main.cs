using System;
using System.Collections.Generic;
using System.Linq;

using Foundation;
using Refractored.XamForms.PullToRefresh.iOS;
using UIKit;
using Lottie.Forms.iOS.Renderers;

namespace RecommendersDemo.iOS
{
    public class Application
    {
        // This is the main entry point of the application.
        static void Main(string[] args)
        {
            // if you want to use a different Application Delegate class from "AppDelegate"
            // you can specify it here.
            UIApplication.Main(args, null, "AppDelegate");
            PullToRefreshLayoutRenderer.Init();
            AnimationViewRenderer.Init();
        }
    }
}
