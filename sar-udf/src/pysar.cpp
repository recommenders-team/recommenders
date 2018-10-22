#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <exception>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <queue>
#include <iostream>

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

namespace py = pybind11;

class MemoryMapFile {
    int _fd;
    void* _addr;
    struct stat _sb;

public:
    MemoryMapFile(std::string& path) : _fd(0), _addr(nullptr) {
        _fd = open(path.c_str(), O_RDONLY);
        if (_fd == -1)
            throw std::domain_error("unable to open file");

        if (fstat(_fd, &_sb) == -1)      
            throw std::domain_error("unable to open file stats");

        _addr = mmap(NULL, _sb.st_size, PROT_READ, MAP_SHARED, _fd, 0);
        if (_addr == MAP_FAILED)
            throw std::domain_error("failed to memory map file");
    }

    ~MemoryMapFile() {
        if (_addr)
            munmap(_addr, _sb.st_size);
        if (_fd > 0)
            close(_fd);
    }

    void* addr() { return _addr; }
};

struct item_score
{
    int32_t id;
    float score;

    int get_id() { return id; }

    float get_score() { return score; }

    struct score_compare {
        bool operator() (const item_score& left, const item_score& right)
        { return left.score > right.score; }
    };

    static struct _id_compare {
        bool operator() (const item_score& left, int32_t right)
        { return left.id < right; }

        bool operator() (int32_t left, const item_score& right)
        { return left < right.id; }
    } id_compare;
};

class SARModel {
    MemoryMapFile _offsets_memory_map;
    MemoryMapFile _related_memory_map;

    int64_t* _offsets;
    item_score* _related;

public:
    SARModel(std::string& offsets_path, std::string& related_path)
        : _offsets_memory_map(offsets_path), _related_memory_map(related_path) { 
        _offsets = (int64_t*)_offsets_memory_map.addr();
        _related = (item_score*)_related_memory_map.addr();
    }

    std::vector<item_score> predict(std::vector<int32_t>& items_of_user, std::vector<float>& ratings, int32_t top_k) {
        std::vector<item_score> preds;

        if (items_of_user.empty())
            return preds;

        std::unordered_set<int32_t> seen_items;
        std::priority_queue<item_score, std::vector<item_score>, item_score::score_compare> top_k_items;

        // loop through items user has seen
        for (auto& iid : items_of_user) {
            // loop through related items
            auto related_beg = _related + _offsets[iid];
            auto related_end = _related + _offsets[iid+1];
            for (;related_beg != related_end; ++related_beg) {
                auto related_item = *related_beg;

                // avoid duplicated
                if (seen_items.find(related_item.id) != seen_items.end())
                    continue;
                seen_items.insert(related_item.id);

                // calculate score
                auto related_item_score = join_prod_sum(items_of_user, ratings, related_item.id);

                if (related_item_score > 0)
                    push_if_better(top_k_items, {related_item.id, related_item_score}, top_k);
            }
        }

        // output top-k items
        while (!top_k_items.empty()) {
            preds.push_back(top_k_items.top());
            top_k_items.pop();
        }

        return preds;
    }

    void push_if_better(std::priority_queue<item_score, std::vector<item_score>, item_score::score_compare>& top_k_items, item_score new_item_score, int32_t top_k) {
        // less than k items
        if ((int32_t)top_k_items.size() < top_k) {
            top_k_items.push(new_item_score);
            return;
        }

        // found a better one?
        if (top_k_items.top().score < new_item_score.score) {
            top_k_items.pop();
            top_k_items.push(new_item_score);
        }
    }

    // join items_of_user with related-related items
    float join_prod_sum(std::vector<int32_t>& items_of_user, std::vector<float>& ratings, int32_t related_item) {
        auto contrib_beg = _related + _offsets[related_item];
        auto contrib_end = _related + _offsets[related_item+1];

        double score = 0;
        auto user_iid = items_of_user.begin();
        auto user_iid_end = items_of_user.end();

        while(true) {
            auto user_iid_v = *user_iid;
            auto& contrib_v = *contrib_beg;

            // binary search
            if (user_iid_v < contrib_v.id) {
                auto user_iid_next = std::lower_bound(user_iid, user_iid_end, contrib_v, item_score::id_compare);
                if (user_iid_next == user_iid_end)
                    break;
                user_iid = user_iid_next;

                continue;
            }

            if(user_iid_v > contrib_v.id) {
                auto contrib_next = std::lower_bound(contrib_beg, contrib_end, user_iid_v, item_score::id_compare);
                if (contrib_next == contrib_end)
                    break;
                contrib_beg = contrib_next;

                continue;
            }

            // got a match!

            // auto score_old = score; // Debug
            score += ratings[&*user_iid - &items_of_user[0]] * contrib_v.score;
            // std::cout << "related_item(" << related_item << ") " << score << " = " << score_old << " + " <<  ratings[&*user_iid - &items_of_user[0]] <<  "  * " << contrib_v.score << std::endl;

            ++user_iid;
            if (user_iid == user_iid_end)
                break;

            ++contrib_beg;
            if (contrib_beg == contrib_end)
                break;
        }

        return score;
    }
};

PYBIND11_MODULE(pysar_cpp, m) {
    py::class_<item_score> sar_pred(m, "SARPrediction");

    sar_pred.def_property_readonly("id", &item_score::get_id);
    sar_pred.def_property_readonly("score", &item_score::get_score);

    py::class_<SARModel> model(m, "SARModelCpp");

    model.def(py::init([](std::string offsets_path, std::string related_path)
                       { return new SARModel(offsets_path, related_path); }))
         .def("predict", &SARModel::predict);
}