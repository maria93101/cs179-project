
#include<iostream>
#include <fstream>
#include<vector>
#include<string>
#include<map>
#include "gpu_data.h"
#include <tuple>
#include<cmath>
#include<algorithm>
#include <chrono>
#include <iterator>
#include <sstream>

using namespace std;
using namespace std::chrono;
Data::Data(void)
{
    
}


/*
 * Returns an average of all the movie ratings a user has seen
 */
float Data::get_avg_rats(int id, bool mu_orient)
{
    vector<float> paired_ratings;
    if (mu_orient) {
        paired_ratings = movie_user[id];
    } else {
        paired_ratings = user_movie[id];
    }
    float summ = 0;
    for (int i = 0; i < paired_ratings.size(); i++)
    {
        summ += paired_ratings[i*3 + 1];
    }
    return summ / (paired_ratings.size()/3);
}

float* Data::movie_user_list(int cur_movie_id){
    vector<float> list = movie_user[cur_movie_id];
    float *arr = new float[list.size()];
    copy(list.begin(), list.end(), arr);
    return arr;
}

int Data::movie_user_list_num(int cur_movie_id){
    vector<float> list = movie_user[cur_movie_id];
    return list.size();
}

vector<float> Data::get_rats_list(int id, bool mu_orient)
{
    vector<float> paired_ratings;
    if (mu_orient) {
        paired_ratings = movie_user[id];
    } else {
        paired_ratings = user_movie[id];
    }
    
    vector<float> ratings;
    for (int i = 0; i < paired_ratings.size(); i++)
    {
        ratings.push_back(paired_ratings[i*3 + 1]);
    }
    return ratings;
}

void Data::read_data(string filename, bool mu)
{
    
    fstream file(filename);
    int uid, mid, date;
    float rating;
    cout << filename<<endl;
    while (file >> uid >> mid >> date >> rating)
    {
        if ( movie_user.find(mid-1) == movie_user.end() ) {
            vector<float> temp;
            movie_user.insert(pair<int, vector<float>>(mid-1, temp));
        }
        movie_user[mid-1].push_back(uid-1);
        movie_user[mid-1].push_back(rating);
        movie_user[mid-1].push_back(date);
        if ( user_movie.find(uid-1) == user_movie.end() ) {
            vector<float> temp;
            user_movie[uid-1]= temp;
        }
        user_movie[uid-1].push_back(mid-1);
        user_movie[uid-1].push_back(rating);
        user_movie[uid-1].push_back(date);
    }
    
    // Assign relevant data to object.
    max_movie_id = 17770;
    max_user_id = 458293;
    cout << "Processing finished"<<endl;
}
