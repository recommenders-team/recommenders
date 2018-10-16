#pragma once

#include <queue>
#include <mutex>
#include <thread>
#include <vector>
#include <map>
#include <stdint.h>
#include <fstream>
#include <atomic>
#include <string>

// pair used to keep track of top-k items
struct item_score
{
    uint32_t iid;
    float score;
};

bool operator<(const item_score& a, const item_score& b);

class SAR
{
    std::vector<uint32_t> _users;
    std::vector<uint32_t> _related_item_offsets;
    std::vector<uint32_t> _related_items;
    std::vector<double> _scores;
    std::map<std::string, uint32_t> _item_to_index_map;
    std::vector<std::string> _item_to_index;

    // for multi-threaded scoring
    std::mutex _predict_mutex;
    std::queue<std::string> _predict_queue;

    uint32_t _top_k;

    // progress bar support
    std::atomic<uint64_t> _predict_progress;
    uint64_t _predict_total;

    void predict_worker();

    void predict_progress();

    uint64_t get_row_count(const char* input_path);

    void predict_single_parquet(const char* input_path);

    void predict(std::string uid, std::vector<uint32_t>& items_of_user, std::vector<double>& ratings, std::ofstream& outfile);

    void push_if_better(std::priority_queue<item_score>& top_k_items, item_score new_item_score, uint32_t top_k);

    float join_prod_sum(std::string& uid, std::vector<uint32_t>& items_of_user, std::vector<double>& ratings, uint32_t related_item);

    uint32_t get_or_insert(std::string& s);

    void index_and_cache(const char* dir_path);

    bool load_from_cache(const char* dir_path);
public:
    // load similarity matrix
    void load(const char* dir_path);

    void predict_parquet(const char* user_to_items_parquet, uint32_t top_k);
};