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
{ return a.score > b.score; }

struct CacheHeader
{
    uint64_t related_item_offsets_count;
    uint64_t related_items_count; // same as scores 
};

void SAR::predict_worker()
{
    try
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

            // cout << "processing " << input_path << endl;
            predict_single_parquet(input_path.c_str());
        }
    }
    catch (std::exception& ex)
    {
        cerr << "Failed: " << ex.what() << endl;
    }
}

void SAR::predict_progress()
{
    cout << endl;

    auto start = chrono::high_resolution_clock::now();
    while (_predict_progress < _predict_total)
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

    cout << endl;
}

uint64_t SAR::get_row_count(const char* input_path)
{
    std::shared_ptr<arrow::io::ReadableFile> infile;
    PARQUET_THROW_NOT_OK(arrow::io::ReadableFile::Open(input_path, arrow::default_memory_pool(), &infile));

    std::unique_ptr<parquet::arrow::FileReader> reader;
    PARQUET_THROW_NOT_OK(parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

    std::shared_ptr<arrow::Array> array;
    PARQUET_THROW_NOT_OK(reader->ReadColumn(0, &array));

    // cout << input_path << ": " << array->length() << endl;

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
    if (!array->type()->Equals(arrow::StringType()))
        throw runtime_error("test input column 0 (user id) must be of type string");
    arrow::StringArray user_id_array(array->data());

    PARQUET_THROW_NOT_OK(reader->ReadColumn(1, &array));
    if (!array->type()->Equals(arrow::StringType()))
        throw runtime_error("test input column 1 (item id) must be of type string");
    arrow::StringArray item_id_array(array->data());

    PARQUET_THROW_NOT_OK(reader->ReadColumn(2, &array));
    if (!array->type()->Equals(arrow::DoubleType()))
        throw runtime_error("test input column 2 (rating) must be of type double");
    arrow::NumericArray<arrow::DoubleType> value_array(array->data());

    auto values = value_array.raw_values();

    // collect all items for a given user (assume sorted by user and item)
    string prev_uid;
    vector<uint32_t> items_of_user;
    vector<double> ratings;

    for (uint64_t idx=0;idx<user_id_array.length();++idx,++values)
    {
        string uid = user_id_array.GetString(idx);

        if (prev_uid != uid)
        {
            predict(prev_uid, items_of_user, ratings, outfile);

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

template<typename T>
void printVector(const char* name, T beg, T end)
{
    cout << name << ": ";
    size_t i = 0;
    for(;beg != end && i < 20;++beg,++i)
        cout << *beg << ",";
    cout << endl;
}

template<typename T>
void printVector(const char* name, vector<T>& vec)
{
    cout << name << ": ";
    for(auto& val : vec)
        cout << val << ",";
    cout << endl;
}

void SAR::predict(string uid, vector<uint32_t>& items_of_user, vector<double>& ratings, ofstream& outfile)
{
    if (items_of_user.empty())
        return;

/*
    if (uid == "496")
    {
        cout << "items_of_user '" << uid << "': ";
        for(auto& val : items_of_user)
            cout << _item_to_index[val] << ",";
        cout << endl << endl;
        for(auto& val : items_of_user)
            cout << val << ",";
        cout << endl;
    }
*/

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

/*
        if (uid == "496")
        {
            cout << "\trelated " << _item_to_index[related_item] << endl;
        }
*/
            // avoid duplicated
            if (seen_items.find(related_item) != seen_items.end())
                continue;
            seen_items.insert(related_item);

            // calculate score
            auto related_item_score = join_prod_sum(uid, items_of_user, ratings, related_item);

            // if (uid == "496" && _item_to_index[related_item] == "590")
                // cout << "related " << _item_to_index[related_item] << ": " << related_item_score << endl;
            if (related_item_score > 0)
                push_if_better(top_k_items, {related_item, related_item_score}, _top_k);
        }
    }

    // output top-k items
    while (!top_k_items.empty())
    {
        auto is = top_k_items.top();
        outfile << uid << "\t" << _item_to_index[is.iid] << "\t" << is.score << endl;

        // if (uid == "496")
            // cout << uid << "\t" << _item_to_index[is.iid] << "\t" << is.score << endl;
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
float SAR::join_prod_sum(string& uid, vector<uint32_t>& items_of_user, vector<double>& ratings, uint32_t related_item) 
{
    auto contrib_beg = &_related_items[0] + _related_item_offsets[related_item];
    auto contrib_end = &_related_items[0] + _related_item_offsets[related_item+1];

    double score = 0;
    auto user_iid = items_of_user.begin();
    auto user_iid_end = items_of_user.end();

/*
    if (uid == "496" && _item_to_index[related_item] == "592")
    {
    contrib_beg = &_related_items[0] + _related_item_offsets[related_item];
    contrib_end = &_related_items[0] + _related_item_offsets[related_item+1];
        cout << "found" << endl;
        for(auto it = contrib_beg;it!=contrib_end;++it)
            cout << _item_to_index[*it] << ","; 
        cout << endl << endl;
        for(auto it = contrib_beg;it!=contrib_end;++it)
            cout << *it << ","; 
        cout << endl;
    }
*/

    while(true)
    {
        auto user_iid_v = *user_iid;
        auto contrib_v = *contrib_beg;

/*
        if (uid == "496" && _item_to_index[related_item] == "592")
        {
                cout << _item_to_index[user_iid_v] << " <-> " << _item_to_index[contrib_v] << endl;
        }
*/
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
/*
        if (uid == "496" && _item_to_index[related_item] == "592")
        {
         cout << "score (" << score << ") += "  << 
             ratings[&*user_iid - &items_of_user[0]] << " * " <<
             _scores[&*contrib_beg - &_related_items[0]] << endl;
        }
*/
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

    // if (score > 0)
        // cout << "score: " << score << endl;

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
    _predict_total = 0;
    _predict_progress = 0;
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
    // num_threads = 1;

    cout << "Thread count: " << num_threads << endl;
    for (int i=0;i<num_threads;i++)
        pool.push_back(std::move(thread(&SAR::predict_worker, this)));

    pool.push_back(std::move(thread(&SAR::predict_progress, this)));

    for (auto& t : pool)
        t.join();
}

void SAR::load(const char* dir_path)
{
    if (load_from_cache(dir_path))
        return;

    index_and_cache(dir_path);
}

class SimilarityFile
{
    std::shared_ptr<arrow::io::ReadableFile> infile;
    std::unique_ptr<parquet::arrow::FileReader> reader;

public:
    SimilarityFile(const char* file_name) : _file_name(file_name)
    {
        PARQUET_THROW_NOT_OK(arrow::io::ReadableFile::Open(file_name, arrow::default_memory_pool(), &infile));

        PARQUET_THROW_NOT_OK( parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));

        std::shared_ptr<arrow::Array> array;
        PARQUET_THROW_NOT_OK(reader->ReadColumn(0, &array));
        if (!array->type()->Equals(arrow::StringType()))
            throw runtime_error("similarity input column 0 (item id 0) must be of type string");
        i1_array = make_unique<arrow::StringArray>(array->data());

        PARQUET_THROW_NOT_OK(reader->ReadColumn(1, &array));
        if (!array->type()->Equals(arrow::StringType()))
            throw runtime_error("similarity input column 1 (item id 1) must be of type string");
        i2_array = make_unique<arrow::StringArray>(array->data());

        PARQUET_THROW_NOT_OK(reader->ReadColumn(2, &array));
        if (!array->type()->Equals(arrow::DoubleType()))
            throw runtime_error("similarity input column 2 (scores) must be of type double");
        value_array = make_unique<arrow::NumericArray<arrow::DoubleType>>(array->data());

        // first_iid = i1_array->GetString(0);
    }

    string _file_name;

    // string first_iid;

    std::unique_ptr<arrow::StringArray> i1_array;
    std::unique_ptr<arrow::StringArray> i2_array;
    std::unique_ptr<arrow::NumericArray<arrow::DoubleType>> value_array;
};

void SAR::index_and_cache(const char* dir_path)
{
    if (!is_directory(dir_path))
    {
        stringstream msg;
        msg << "Path '" << dir_path << "' not found or not a directory.";
        throw std::runtime_error(msg.str());
    }

    cout << "Mapping item ids to indicies..." << endl;

    vector<SimilarityFile> sim_files;
    vector<string> files;

    directory_iterator end_itr;
    for (directory_iterator itr(dir_path); itr != end_itr; ++itr)
    {
        string file_name = itr->path().native();
        string extension = ".parquet";
        if (is_directory(itr->status()) || !std::equal(extension.rbegin(), extension.rend(), file_name.rbegin()))
            continue;

        files.push_back(file_name);
        sim_files.push_back(std::move(SimilarityFile(file_name.c_str())));
        // cout << "\t" << file_name << endl;
    }

/*
    sort(sim_files.begin(), sim_files.end(), 
        [](const auto & a, const auto & b) -> bool
    { 
        return a.first_iid < b.first_iid; 
    });
*/
    cout << "Building continuous index..." << endl;
    string prev_iid_str;
    uint32_t i1_assumption = 0;

    for (auto& sim_file : sim_files)
    {
        cout << "file " << sim_file._file_name << endl;
        for (uint64_t idx = 0;idx<sim_file.i1_array->length();++idx)
        {
            auto s1 = sim_file.i1_array->GetString(idx);

            if (s1 == prev_iid_str)
                continue;

            auto i1 = get_or_insert(s1);

            if (i1 != i1_assumption)
            {
                cout << "failed assumption.3 " << s1 << " -> " << i1 << " vs " << i1_assumption << " line nr " << idx << endl;
                exit(-1);
            }

            i1_assumption++;
            // cout << "x: " << x << " to " << s1 << endl;

            prev_iid_str = s1;
        }
    }
 
    // string x463 = "463";
    // cout << "463 -> " << get_or_insert(x463) << endl;

    cout << "Building lookup..." << endl;
    uint32_t prev_iid = 0xFFFFFFFF; 
    for (auto& sim_file : sim_files)
    {
        const double* values = sim_file.value_array->raw_values();
        for (uint64_t idx = 0;idx<sim_file.i1_array->length();++idx,++values)
        {
                // cout << "idx " << idx << endl;
            auto s1 = sim_file.i1_array->GetString(idx);
            auto s2 = sim_file.i2_array->GetString(idx);

            auto i1 =  get_or_insert(s1);
            auto i2 =  get_or_insert(s2);

            // expect ORDERY BY uid, iid
            if (prev_iid != i1)
            {
                if (i1 != _related_item_offsets.size())
                {
                    cout << "i1 mismatch " << s1 << " -> " << i1 << " vs " << _related_item_offsets.size() << endl;
                    exit(-2);
                }

                _related_item_offsets.push_back(_related_items.size());
                prev_iid = i1;   
            }
            
            _related_items.push_back(i2);
            _scores.push_back(*values);
        }
    }
    // final element to allow iid+1
    _related_item_offsets.push_back(_related_items.size());

    {
        cout << "Saving cache..." << endl;

        stringstream cache_path;
        cache_path << dir_path << "/cache.bin";

        ofstream of(cache_path.str().c_str(), ios::out|ios::binary);
        CacheHeader header = { _related_item_offsets.size(), _related_items.size() };
        of.write((const char*)&header, sizeof(header));
        of.write((const char*)&_related_item_offsets[0], sizeof(uint32_t) * header.related_item_offsets_count); 
        of.write((const char*)&_related_items[0], sizeof(uint32_t) * header.related_items_count); 
        of.write((const char*)&_scores[0], sizeof(double) * header.related_items_count); 

        // keep original ids
        stringstream id_path;
        id_path << dir_path << "/cache.ids";

        ofstream of_ids(id_path.str().c_str(), ios::out);
        for(auto& s : _item_to_index)
            of_ids << s << endl;
    }
}

bool SAR::load_from_cache(const char* dir_path)
{
    cout << "Loading cache..." << endl;

    stringstream cache_path;
    cache_path << dir_path << "/cache.bin";

    if (!exists(cache_path.str().c_str()))
        return false;

    file_mapping mapping(cache_path.str().c_str(), read_only);
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

    stringstream id_path;
    id_path << dir_path << "/cache.ids";

    // load mapping
    uint32_t id = 0;
    ifstream if_ids(id_path.str().c_str());
    string line;

    while (getline(if_ids, line))
    {
        _item_to_index_map.insert(make_pair(line, id++));
        _item_to_index.push_back(line);
    }

    return true;
}