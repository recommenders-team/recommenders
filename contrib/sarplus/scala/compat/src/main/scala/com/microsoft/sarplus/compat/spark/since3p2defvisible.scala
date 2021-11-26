package com.microsoft.sarplus.spark

import scala.annotation.{StaticAnnotation, compileTimeOnly}
import scala.language.experimental.macros
import scala.reflect.macros.Context

import util.Properties.versionNumberString

@compileTimeOnly("enable macro paradise to expand macro annotations")
class since3p2defvisible extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro since3p2defvisibleMacro.impl
}

object since3p2defvisibleMacro {
  def impl(c: Context)(annottees: c.Tree*) = {
    import c.universe._
    annottees match {
      case q"$mods def $name[..$tparams](...$paramss): $tpt = $body" :: tail =>
        // NOTE: There seems no way to find out the Spark version.
        if (versionNumberString.startsWith("2.12.14")) {
          q"""
            $mods def $name[..$tparams](...$paramss): $tpt =
              $body
          """
        } else {
          q""
        }
      case _ => throw new IllegalArgumentException("Please annotate a method")
    }
  }
}
