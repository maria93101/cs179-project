#include<iostream>
#include <fstream>
#include<vector>
#include<string>
#include<map>
#include "data.h"
#include <tuple>
#include<cmath>
#include<algorithm>
#include "knn.cuh"
#include <iterator>
#include <sstream>
#include <chrono>
//Single line comment
using namespace std;
using namespace std::chrono;

// This gives us the movie ratings for the 2 movies by the same user.
vector<float> get_paired_user_ratings(vector<float> movie_i,
                                      vector<float> movie_j, bool first)
{
    sort(movie_i.begin(), movie_i.end());
    sort(movie_j.begin(), movie_j.end());
    int counter_i = 0;
    int counter_j = 0;
    vector<float> res;
    while(counter_i < movie_i.size()/3 and counter_j < movie_j.size()/3)
    {
        //
        if ((movie_i[counter_i*3]) == (movie_j[counter_j*3]))
        {
            if ((movie_j[counter_j*3 + 1])!= 0 and 0 != (movie_i[1+3*counter_i]))
            {
                if (first)
                {
                    res.push_back(movie_i[1+3*counter_i]);
                }
                else{
                    res.push_back(movie_j[1+3*counter_j]);
                    
                }
            }
            counter_i ++;
            counter_j ++;
        }
        else {
            if ((movie_i[3*counter_i]) > (movie_j[3*counter_j]))
            {
                counter_j ++;
            }
            else {
                counter_i ++;
            }
        }
    }
    return res;
}

float pearson(vector<float> item_rats_i, vector<float> item_rats_j)
{
    float L;
    float top = 0, bottom = 0;
	int size_i = item_rats_i.size();
    int size_j = item_rats_j.size();
    float *item_i_diff = new float[size_i];
    float *item_j_diff = new float[size_j];
    float i_sum = 0, j_sum = 0;
    L = item_rats_i.size();
    if (L <= 1)
    {
        return 0;
    }
    for (int i = 0; i < L; i++)
    {
        i_sum += item_rats_i[i];
        j_sum += item_rats_j[i];
    }
    float x_i_mean = i_sum / L;
    float x_j_mean = j_sum / L;
    float MSE_i = 0;
    float MSE_j = 0;
    
    for(int i = 0; i < L; i++)
    {
        item_i_diff[i] = item_rats_i[i] - x_i_mean;
        item_j_diff[i] = item_rats_j[i] - x_j_mean;
        MSE_i += pow(item_i_diff[i], 2);
        MSE_j += pow(item_j_diff[i], 2);
    }
    for (int i = 0; i < L; i++)
    {
		float diff_i = item_i_diff[i];
        float diff_j = item_j_diff[i];
        top += diff_i * diff_j;
    }
    
    top *= 1/(L-1);
    bottom = sqrt(1/(L-1) * MSE_i)*sqrt(1/(L-1) * MSE_j);
    
    if (bottom == 0)
    {
        return 0;
    }
	free(item_i_diff);
	free(item_j_diff);
    return top/bottom;

}

float get_cij(vector<float> item_rats_i, vector<float> item_rats_j, int alpha)
{
    return pearson(item_rats_i, item_rats_j)*item_rats_i.size()/(item_rats_i.size()+alpha);
}

void knn(Data *data, float **cij_lib, int alpha, int k, string ifile, string ofile, bool probe)
{
    vector<float> true_rat;
    int counter = 0;
    
    ofstream txtOut;
    txtOut.open (ofile);
    vector<pair<float, float>> paired_movieid;
    vector<int> val_uid;
    vector<int> val_mid;
    vector<int> val_ratings;
    ifstream file(ifile);
    int uid, mid, date, rating;
    float e_out = 0, corr_e_out=0;
    if (probe)
    {
        while (file >> uid >> mid >> date >> rating) {
            val_uid.push_back(uid-1);
            val_mid.push_back(mid-1);
            val_ratings.push_back(rating);
        }
    }
    else
    {
        while (file >> uid >> mid >> date) {
            val_uid.push_back(uid-1);
            val_mid.push_back(mid-1);
            val_ratings.push_back(0);
        }
    }
    for (int i = 0; i < val_uid.size(); i++)
    {
        int user_id = val_uid[i];
        int cur_movie_id = val_mid[i];
        vector<float> user_list_movies = data->user_movie[user_id];
        
        float * cij = new float[user_list_movies.size()/3];
		float * cijr = new float[user_list_movies.size()/3];
		vector<pair<float, float>> cijvec;
        
        //Using pearson's correlation to get a list of correlations for
        // the list of movies.
		////////////////////////////////////////////////////////
        for (int j = 0; j < user_list_movies.size()/3; j++)
        {
            int id = user_list_movies[3*j];
            
            if (id < cur_movie_id)
            {
                int temp = id; id = cur_movie_id; cur_movie_id = temp;
            }
            if (cij_lib[cur_movie_id][id] == 0)
            {
                vector <float> movie_rat_i = get_paired_user_ratings( data->movie_user[cur_movie_id], data->movie_user[id], true);
                vector <float> movie_rat_j = get_paired_user_ratings( data->movie_user[cur_movie_id], data->movie_user[id], false);
                cij_lib[cur_movie_id][id] = get_cij(movie_rat_i, movie_rat_j, alpha);
            }
			cij[j] = (cij_lib[cur_movie_id][id]);
			cijr[j] = (cij_lib[cur_movie_id][id]*user_list_movies[3*j+1]);
        }
        // Calling the merge sort kernal to sort cij, and cijr along with it.
		callMergeKernel(128, 128, cij, cijr, user_list_movies.size()/3);
        float top = 0, bottom = 0;
        int loops = user_list_movies.size()/3 > k ? k : user_list_movies.size()/3;
        
        // Calling the summing kernel to get the sums.
        top = correlationKernelSum(cijr, loops, user_list_movies.size()/3);
        bottom = correlationKernelSum(cij, loops, user_list_movies.size()/3);
        float r_hat = bottom == 0? 3 : top/bottom;
        e_out += pow((val_ratings[i] - r_hat), 2);

    }
    e_out = pow(e_out / val_uid.size(), 0.5) ;
    cout << "error: "<<e_out<<endl;
    txtOut.close();
}

//This is where the execution of program begins
int main()
{
    Data data;
    
    data.read_data("resource/small_train.txt", true);
    
    string ifile = "resource/small_probe.txt";
    string ofile = "resource/small_knn_probe.txt";
    
    float **cij_lib = new float*[17770];
    for (int i = 0; i < 17770; i++)
    {
        cij_lib[i] = new float[17770];
        for (int j = 0; j < 17770; j++)
        {
            cij_lib[i][j] = 0;
        }
    }
    int alpha = 300;
    int k = 30;
    auto start = high_resolution_clock::now();
    
    knn(&data, cij_lib, alpha, k, ifile, ofile, true);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start);
    cout << "duration: "<< duration.count() <<endl;
    for(int i = 0; i < 17770; ++i) {
        delete [] cij_lib[i];
    }
    delete [] cij_lib;
    
    return 0;
}
