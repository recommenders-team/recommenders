#include "SAR.h"
#include <iostream>

int main(int argc, char *argv[]) 
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << argv[0] << " <similarity-parquet-directory> <test-user-with-train-items-parquet-directory> <top-k>" << std::endl;
        return -1;
    }

    SAR sar;

    // const char* dir_path = "/mnt/c/Data/netflix/similarity-full.parquet";
    // "/mnt/c/Data/netflix/test_users_with_train_items.parquet"
    sar.load(argv[1]);
    sar.predict_parquet(argv[2], atoi(argv[3]));

    return 0;
}