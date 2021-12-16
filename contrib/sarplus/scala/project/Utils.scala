object Utils {
  def spark32classifer(classifier: Option[String], sparkVer: String) = {
    classifier match {
      case None => {
        val splitVer = sparkVer.split('.')
        val major = splitVer(0).toInt
        val minor = splitVer(1).toInt
        if (major >=3 && minor >= 2) Some("spark32") else None
      }
      case Some(s: String) => Some(s)
    }
  }
}
