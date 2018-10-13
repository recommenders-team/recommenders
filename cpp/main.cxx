// On my way back from Boston to Salzburg (10/13/2018)

#include <iostream>
#include <memory>
#include <queue>

#include <string>
#include <fstream>
#include <sstream>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#include <fstream>
#include <iostream>
#include <set>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#include <parquet/exception.h>

using namespace std;
using namespace boost;

struct item_score
{
    uint32_t iid;
    float score;
};

bool operator<(const item_score& a, const item_score& b)
{ return a.score < b.score; }

class SAR
{
    // TODO: uint/ulong?
    vector<uint32_t> _users;
    vector<uint32_t> _relatedItemOffsets;
    vector<uint32_t> _relatedItems;
    vector<double> _scores;

public:

    void load_csv(const char* similarity_csv)
    {
        ifstream infile(similarity_csv);

        boost::char_separator<char> sep(",", "", boost::keep_empty_tokens);
        uint64_t line_nr = 0;

        std::string line;
        // TODO: needs to initialized with something
        uint32_t prev_iid = 0xFFFFFFF; 

        while (getline(infile, line))
        {
            //try
            {
                boost::tokenizer< boost::char_separator<char> > tok(line, sep);
                auto beg = tok.begin();

                auto i1 = lexical_cast<uint32_t>(*beg); ++beg;
                auto i2 = lexical_cast<uint32_t>(*beg); ++beg;
                auto score = lexical_cast<uint32_t>(*beg);
                // cout << i1 << " " << i2 << " " << score << endl;

                // expect ORDERY BY uid, iid
                if (prev_iid != i1)
                {
                    // TODO: off-by-1?
                    _relatedItemOffsets.push_back(_relatedItems.size());
                    prev_iid = i1;   
                }
                
                _relatedItems.push_back(i2);
                _scores.push_back(score);
            }
            // final 
            /*
            catch(...)
            {
                cout << line << endl;
                return;
            }
            */
        }
        _relatedItemOffsets.push_back(_relatedItems.size());

        // printVector("relatedItemOffset", _relatedItemOffsets);
        // printVector("relatedItems", _relatedItems);
        // printVector("scores", _scores);

        // print matrix...
        // header
        cout << "  | "; 
        for (uint32_t col = 0; col <= prev_iid; ++col)
            cout << col << " ";
        cout << endl;
        cout << "----";
        for (uint32_t col = 0; col <= prev_iid; ++col)
            cout << "--";
        cout << endl;

        for (uint32_t row = 0; row <= prev_iid; ++row)
        {
            cout << row << " | ";

            auto related_beg = &_relatedItems[0] + _relatedItemOffsets[row];
            auto related_end = &_relatedItems[0] + _relatedItemOffsets[row+1];

            uint32_t col = 0;
            for (;related_beg != related_end; ++related_beg)
            {
                auto related_item = *related_beg;
                for (;col < related_item;++col)
                    cout << "  ";
                ++col;
                auto score = _scores[&*related_beg - &_relatedItems[0]];
                cout << score << " ";
            }
            cout << endl;
        }        
        cout << endl;
    }

    void predict(const char* user_to_items_csv, uint32_t top_k, const char* output)
    {
        ifstream infile(user_to_items_csv);
        ofstream outfile(output);

        boost::char_separator<char> sep(",", "", boost::keep_empty_tokens);
        uint64_t line_nr = 0;

        std::string line;
        vector<uint32_t> items_of_user;
        vector<uint32_t> ratings;

        uint32_t prev_uid = 0xFFFFF;

        while (getline(infile, line))
        {
            boost::tokenizer< boost::char_separator<char> > tok(line, sep);
            auto beg = tok.begin();

            auto uid = lexical_cast<uint32_t>(*beg); ++beg;
            auto iid = lexical_cast<uint32_t>(*beg); ++beg;
            auto rating = lexical_cast<uint32_t>(*beg);
            // cout << uid << " " << iid << " " << rating << endl;

            if (prev_uid != uid)
            {
                predict(items_of_user, ratings, top_k, outfile);

                items_of_user.clear();
                ratings.clear();
                prev_uid = uid;
            }

            // nested?
            items_of_user.push_back(iid);           
            ratings.push_back(rating);
        }

        predict(items_of_user, ratings, top_k, outfile);
    }

    void predict(vector<uint32_t>& items_of_user, vector<uint32_t>& ratings, uint32_t top_k, ofstream& outfile)
    {
        if (items_of_user.empty())
            return;

        printVector("user", items_of_user);

        // TODO: avoid allocation
        // bitset? maybe not good as we need to zero everything? SSE?
        set<uint32_t> seen_items;
        priority_queue<item_score> top_k_items;

        // loop through items user has seen
        for (auto& iid : items_of_user)
        {
            // loop through related items
            auto related_beg = &_relatedItems[0] + _relatedItemOffsets[iid];
            auto related_end = &_relatedItems[0] + _relatedItemOffsets[iid+1];
            for (;related_beg != related_end; ++related_beg)
            {
                auto related_item = *related_beg;

                // avoid duplicated
                if (seen_items.find(related_item) != seen_items.end())
                {
                    // cout << "skip " << related_item << endl;
                    continue;
                }
                seen_items.insert(related_item);

                // calculate score
                auto related_item_score = join_prod_sum(items_of_user, ratings, related_item);

                push_if_better(top_k_items, {related_item, related_item_score}, top_k);
            }
        }

        while (!top_k_items.empty())
        {
            // TODO: optimize... just iterate?
            auto is = top_k_items.top();
            cout << "\t" << is.iid << ": " << is.score << endl;
            top_k_items.pop();
        }
    }

    void push_if_better(priority_queue<item_score>& top_k_items, item_score new_item_score, uint32_t top_k)
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
            // TODO: we can do inplace pop/push
            top_k_items.pop();
            top_k_items.push(new_item_score);
        }
    }

/*
    uint32_t* binary_search(
        uint32_t item,
        uint32_t* b,
        uint32_t* b_end)
    {
        // a,b,c,m,n,x,y
        // a,m,z

        // b < m --> search for m between c and y
        while (true)
        {
            uint32_t* b_mid;

            // TODO: das ist sicher falsch
            if (*b < item)
                b_mid = b + (b_end - b) / 2
            else if(*b > item)
                b_end = b + (b_end - b) / 2
            else
                // match
                return b;
        }

        return nullptr;
    }
*/

    // join items_of_user with related-related items
    float join_prod_sum(vector<uint32_t>& items_of_user, vector<uint32_t>& ratings, uint32_t related_item) 
    {
        auto contrib_beg = &_relatedItems[0] + _relatedItemOffsets[related_item];
        auto contrib_end = &_relatedItems[0] + _relatedItemOffsets[related_item+1];

        // linear scan assuming sorted items
        // TODO: optimize using binary search
        float score = 0;
        auto user_iid = items_of_user.begin();
        while(true)
        {
            if (*user_iid < *contrib_beg)
            {
                ++user_iid;
                if (user_iid == items_of_user.end())
                    break;
            }
            else if(*user_iid > *contrib_beg)
            {
                ++contrib_beg;
                if (contrib_beg == contrib_end)
                    break;
            }
            else
            {
                // match
                score += ratings[&*user_iid - &items_of_user[0]]
                    * _scores[&*contrib_beg - &_relatedItems[0]];

                ++user_iid;
                if (user_iid == items_of_user.end())
                    break;

                ++contrib_beg;
                if (contrib_beg == contrib_end)
                    break;
            }
        }

        return score;
    }

    template<typename T>
    void printVector(const char* name, vector<T>& vec)
    {
        cout << name << ": ";
        for(auto& val : vec)
            cout << val << ",";
        cout << endl;
    }

    void load()
    {
         std::cout << "Reading first RowGroup of parquet-arrow-example.parquet" << std::endl;
  std::shared_ptr<arrow::io::ReadableFile> infile;
  PARQUET_THROW_NOT_OK(arrow::io::ReadableFile::Open(
      "/mnt/c/work/Recommenders/SQLWH/parsed_netflix.fastparquet",
      arrow::default_memory_pool(), &infile));

  std::unique_ptr<parquet::arrow::FileReader> reader;
  PARQUET_THROW_NOT_OK(
      parquet::arrow::OpenFile(infile, arrow::default_memory_pool(), &reader));
  std::shared_ptr<arrow::Table> table;
  PARQUET_THROW_NOT_OK(reader->RowGroup(0)->ReadTable(&table));
  std::cout << "Loaded " << table->num_rows() << " rows in " << table->num_columns()
            << " columns." << std::endl;
        // load into members
    }
};

int main() 
{
    SAR sar;

    // works!
    // sar.load();

    // read similarity matrix
    sar.load_csv("similarity.csv");

    sar.predict("test.csv", 2, "predict.csv");

    // TODO: thread parallel predict


    return 0;
}