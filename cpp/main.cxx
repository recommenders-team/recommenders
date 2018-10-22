#include "SAR.h"
#include <iostream>

int main(int argc, char *argv[]) 
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <similarity-parquet-directory> <test-user-with-train-items-parquet-directory> <top-k>" << std::endl;
        return -1;
    }

    try
    {
        SAR sar;

        sar.load(argv[1]);
        sar.predict_parquet(argv[2], atoi(argv[3]));

        return 0;
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
        return -1;
    }
}