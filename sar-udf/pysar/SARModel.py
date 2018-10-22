import pysar_cpp
import os


class SARModel:
    def __init__(self, path):
        # find the .sar.related & .sar.offsets files
        all_files = os.listdir(path)

        def find_or_raise(extension):
            files = [f for f in all_files if f.endswith(extension)]
            if len(files) != 1:
                raise ValueError(
                    "Directory '%s' must contain exactly 1 file ending in '%s'"
                    % (path, extension)
                )
            return path + "/" + files[0]

        # instantiate C++ backend
        self.model = pysar_cpp.SARModelCpp(
            find_or_raise(".sar.offsets"), find_or_raise(".sar.related")
        )

    def predict(self, items, ratings, top_k):
        return self.model.predict(items, ratings, top_k)

