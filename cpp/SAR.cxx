#include "SAR.h"

#include <iostream>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <unordered_set>
#include <chrono> 

#include <sstream>
#include <set>
#include <map>

#include <boost/filesystem.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/exception.h>

using namespace boost;
using namespace boost::filesystem;
using namespace boost::interprocess;

using namespace std;

bool operator<(const item_score& a, const item_score& b)
{ return a.score < b.score; }

struct CacheHeader
{
    uint64_t related_item_offsets_count;
    uint64_t related_items_count; // same as scores 
};

void SAR::predict_worker()
{
    while (true)
    {
        string input_path;

        {
            unique_lock<mutex> lock(_predict_mutex);
            if (_predict_queue.empty())
                return;

            input_path = _predict_queue.front();
            _predict_queue.pop();
        }

        predict_single_parquet(input_path.c_str());
    }
}

void SAR::predict_progress()
{
    cout << endl << endl;

    auto start = chrono::high_resolution_clock::now();
    while (true)
    {
        using namespace std::chrono_literals;

        this_thread::sleep_for(2s);

        auto now = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = now - start; // default is seconds

        double progress = _predict_progress;

        double expected_time = (_predict_total - progress) * (elapsed.count() / progress);

        stringstream msg;
        msg << "\r"
            << setw(9) << (uint64_t)progress << "/" << _predict_total << " ("
            << fixed << setprecision(1) << setw(4) << (100 * progress / _predict_total) << "%) "
            << "Estimated time left ";
    
        if (expected_time > 3600)
            msg << (expected_time / 3600) << "h";
        else if (expected_time > 60)
            msg << (expected_time / 60) << "min";
        else
            msg << expected_time << "sec";

        msg << "      \r";

        cout << msg.str() << flush;
    }
}

uint64_t SAR::get_row_count(const char* input_path)
{
    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_THROW_NOT_OK(arrow::io::ReadableFile::Open(input_path, arrow::default_memory_pool(), &infile));

    std::unique_ptr<parquet::arrow::FileReader> reader;
    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

    std::shared_ptr<arrow::Array> array;
    PARQUET_THROW_NOT_OK(reader->ReadColumn(0, &array));

    return array->length();
}

void SAR::predict_single_parquet(const char* input_path)
{
    stringstream output_path;
    output_path << input_path << ".predict.csv";
    
    ofstream outfile(output_path.str().c_str());

    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_THROW_NOT_OK(arrow::io::ReadableFile::Open(input_path, arrow::default_memory_pool(), &infile));

    std::unique_ptr<parquet::arrow::FileReader> reader;
    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

    std::shared_ptr<arrow::Array> array;

    PARQUET_THROW_NOT_OK(reader->ReadColumn(0, &array));
    arrow::StringArray user_id_array(array->data());

    PARQUET_THROW_NOT_OK(reader->ReadColumn(1, &array));
    arrow::StringArray item_id_array(array->data());

    PARQUET_THROW_NOT_OK(reader->ReadColumn(2, &array));
    arrow::NumericArray<arrow::DoubleType> value_array(array->data());
    // cout << "Rating type: " << array->type()->ToString() << endl;
    // arrow::NumericArray<arrow::Int64Type> value_array(array->data());

    auto values = value_array.raw_values();

    // collect all items for a given user (assume sorted by user and item)
    string prev_uid;
    vector<uint32_t> items_of_user;
    vector<double> ratings;

    for (uint64_t idx = 0;idx<user_id_array.length();++idx,++values)
    {
        string uid = user_id_array.GetString(idx);

        if (prev_uid != uid)
        {
            predict(uid, items_of_user, ratings, outfile);

            items_of_user.clear();
            ratings.clear();
            prev_uid = uid;
        }

        string iid_str = item_id_array.GetString(idx);
        uint32_t iid = _item_to_index_map.find(iid_str)->second;

        items_of_user.push_back(iid);           
        ratings.push_back((double)*values);
    }

    predict(prev_uid, items_of_user, ratings, outfile);
}

void SAR::predict(string uid, vector<uint32_t>& items_of_user, vector<double>& ratings, ofstream& outfile)
{
    if (items_of_user.empty())
        return;

    unordered_set<uint32_t> seen_items;
    priority_queue<item_score> top_k_items;

    // loop through items user has seen
    for (auto& iid : items_of_user)
    {
        // loop through related items
        auto related_beg = &_related_items[0] + _related_item_offsets[iid];
        auto related_end = &_related_items[0] + _related_item_offsets[iid+1];
        for (;related_beg != related_end; ++related_beg)
        {
            auto related_item = *related_beg;

            // avoid duplicated
            if (seen_items.find(related_item) != seen_items.end())
                continue;
            seen_items.insert(related_item);

            // calculate score
            auto related_item_score = join_prod_sum(items_of_user, ratings, related_item);

            push_if_better(top_k_items, {related_item, related_item_score}, _top_k);
        }
    }

    // output top-k items
    while (!top_k_items.empty())
    {
        auto is = top_k_items.top();
        outfile << uid << "\t" << _item_to_index[is.iid] << "\t" << is.score << endl;
        top_k_items.pop();
    }

    // counts the user-item-pairs
    _predict_progress += items_of_user.size();
}

void SAR::push_if_better(priority_queue<item_score>& top_k_items, item_score new_item_score, uint32_t top_k)
{
    // less than k items
    if (top_k_items.size() < top_k)
    {
        top_k_items.push(new_item_score);
        return;
    }

    // found a better one?
    if (top_k_items.top().score < new_item_score.score)
    {
        top_k_items.pop();
        top_k_items.push(new_item_score);
    }
}

// join items_of_user with related-related items
float SAR::join_prod_sum(vector<uint32_t>& items_of_user, vector<double>& ratings, uint32_t related_item) 
{
    auto contrib_beg = &_related_items[0] + _related_item_offsets[related_item];
    auto contrib_end = &_related_items[0] + _related_item_offsets[related_item+1];

    double score = 0;
    auto user_iid = items_of_user.begin();
    auto user_iid_end = items_of_user.end();

    while(true)
    {
        auto user_iid_v = *user_iid;
        auto contrib_v = *contrib_beg;

        // binary search
        if (user_iid_v < contrib_v)
        {
            auto user_iid_next = lower_bound(user_iid, user_iid_end, contrib_v);
            if (user_iid_next == user_iid_end)
                break;
            user_iid = user_iid_next;

            continue;
        }

        if(user_iid_v > contrib_v)
        {
            auto contrib_next = lower_bound(contrib_beg, contrib_end, user_iid_v);
            if (contrib_next == contrib_end)
                break;
            contrib_beg = contrib_next;

            continue;
        }

        // match
        score += ratings[&*user_iid - &items_of_user[0]]
            * _scores[&*contrib_beg - &_related_items[0]];

        ++user_iid;
        if (user_iid == user_iid_end)
            break;

        ++contrib_beg;
        if (contrib_beg == contrib_end)
            break;
    }

    return score;
}

// helper to remap input id-strings to continuous numbers
uint32_t SAR::get_or_insert(string& s)
{
    auto it = _item_to_index_map.find(s);
    if (it != _item_to_index_map.end())
        return it->second;

    auto id = _item_to_index_map.size();
    _item_to_index_map.insert(make_pair(s, id));
    _item_to_index.push_back(s);

    return id;
}

void SAR::predict_parquet(const char* user_to_items_parquet, uint32_t top_k)
{
    _top_k = top_k;

    directory_iterator end_itr;
    for (directory_iterator itr(user_to_items_parquet); itr != end_itr; ++itr)
    {
        string file_name = itr->path().native();
        string extension = ".parquet";
        if (is_directory(itr->status()) || !std::equal(extension.rbegin(), extension.rend(), file_name.rbegin()))
            continue;

        _predict_total += get_row_count(file_name.c_str());
        _predict_queue.push(file_name);
    }

    vector<thread> pool;
    int num_threads = thread::hardware_concurrency();

    cout << "Thread count: " << num_threads << endl;
    for (int i=0;i<num_threads;i++)
        pool.push_back(thread(&SAR::predict_worker, this));

    pool.push_back(thread(&SAR::predict_progress, this));

    for (int i=0;i<num_threads;i++)
        pool[i].join();
}

void SAR::load(const char* dir_path)
{
    if (exists("cache.bin"))
    {
        load_from_cache();
        return;
    }

    load_and_cache(dir_path);
}

void SAR::load_and_cache(const char* dir_path)
{
    if (!exists(dir_path))
    {
        cerr << "Path '" << dir_path << "' not found";
        return;
    }

    uint32_t prev_iid = 0xFFFFFFFF; 

    directory_iterator end_itr;
    for (directory_iterator itr(dir_path); itr != end_itr; ++itr)
    {
        string file_name = itr->path().native();
        string extension = ".parquet";
        if (is_directory(itr->status()) || !std::equal(extension.rbegin(), extension.rend(), file_name.rbegin()))
            continue;

        cout << "Reading " << file_name << endl;
        std::shared_ptr<arrow::io::ReadableFile> infile;
        PARQUET_THROW_NOT_OK(arrow::io::ReadableFile::Open( file_name.c_str(), arrow::default_memory_pool(), &infile));

        std::unique_ptr<parquet::arrow::FileReader> reader;
        PARQUET_THROW_NOT_OK( parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

        std::shared_ptr<arrow::Array> array;

        PARQUET_THROW_NOT_OK(reader->ReadColumn(0, &array));
        arrow::StringArray i1_array(array->data());

        PARQUET_THROW_NOT_OK(reader->ReadColumn(1, &array));
        arrow::StringArray i2_array(array->data());

        PARQUET_THROW_NOT_OK(reader->ReadColumn(2, &array));
        arrow::NumericArray<arrow::DoubleType> value_array(array->data());

        const double* values = value_array.raw_values();
        for (uint64_t idx = 0;idx<i1_array.length();++idx,++values)
        {
            auto s1 = i1_array.GetString(idx);
            auto s2 = i2_array.GetString(idx);

            auto i1 =  get_or_insert(s1);
            auto i2 =  get_or_insert(s2);

            // expect ORDERY BY uid, iid
            if (prev_iid != i1)
            {
                cout << "item " << i1 << " has " << _related_items.size() << endl;
                _related_item_offsets.push_back(_related_items.size());
                prev_iid = i1;   
            }
            
            _related_items.push_back(i2);
            _scores.push_back(*values);
        }
        _related_item_offsets.push_back(_related_items.size());
    }

    {
        // cache data
        ofstream of("cache.bin", ios::out|ios::binary);
        CacheHeader header = { _related_item_offsets.size(), _related_items.size() };
        of.write((const char*)&header, sizeof(header));
        of.write((const char*)&_related_item_offsets[0], sizeof(uint32_t) * header.related_item_offsets_count); 
        of.write((const char*)&_related_items[0], sizeof(uint32_t) * header.related_items_count); 
        of.write((const char*)&_scores[0], sizeof(double) * header.related_items_count); 

        // keep original ids
        ofstream of_ids("cache.ids", ios::out);
        for(auto& s : _item_to_index)
            of_ids << s << endl;
    }
}

void SAR::load_from_cache()
{
    cout << "Loading cache..." << endl;

    file_mapping mapping("cache.bin", read_only);
    mapped_region mapped_rgn(mapping, read_only);

    const char* data = static_cast<const char*>(mapped_rgn.get_address());

    CacheHeader* header = (CacheHeader*)data;

    data += sizeof(CacheHeader);
    const char* data_end = data + sizeof(uint32_t) * header->related_item_offsets_count;
    _related_item_offsets = vector<uint32_t>((uint32_t*)data, (uint32_t*)data_end);

    data = data_end;
    data_end = data + sizeof(uint32_t) * header->related_items_count;
    _related_items = vector<uint32_t>((uint32_t*)data, (uint32_t*)data_end);

    data = data_end;
    data_end = data + sizeof(double) * header->related_items_count;
    _scores = vector<double>((double*)data, (double*)data_end);

    cout << "Similarity matrix number of non-zero items: " << _related_items.size() << endl;

    // load mapping
    uint32_t id = 0;
    ifstream if_ids("cache.ids");
    string line;

    while (getline(if_ids, line))
    {
        _item_to_index_map.insert(make_pair(line, id++));
        _item_to_index.push_back(line);
    }
}