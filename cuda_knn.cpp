
#include<iostream>
#include <fstream>
#include<vector>
#include<string>
#include<map>
#include "data.h"
#include <tuple>
#include<cmath>
#include<algorithm>
#include <cuda_runtime.h>
#include "knn.cuh"
#include <iterator>
#include <sstream>
#include <chrono>
#include <string.h>
//Single line comment
using namespace std;
using namespace std::chrono;

#define gpu_errchk(ans) { gpu_assert((ans), __FILE__, __LINE__); }
inline void gpu_assert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "gpu_assert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        exit(code);
    }
}


// It takes about 5 minutes to run..

void organize(Data *data, float **data_lib, int cur_movie_id, vector<float> user_list_movies, int *max_data_size)
{
	
	int max_size = data->movie_user[cur_movie_id].size();


 	for (int j = 0; j < user_list_movies.size()/3; j++)
	{
		int id = user_list_movies[3*j];
		vector <float> movie_rat = data->movie_user[id];
		max_size = max_size < movie_rat.size() ? movie_rat.size() : max_size;
	}
	*data_lib = new float[  (user_list_movies.size()/3+1) * max_size];//(float *) malloc(sizeof (float) * (user_list_movies.size()/3+1) * max_size);
	cout << "emmm"<<endl;
	memset(*data_lib, -1, sizeof(float)*(user_list_movies.size()/3+1)*max_size);
    for (int j = 0; j < user_list_movies.size()/3; j++)
	{
        int id = user_list_movies[3*j];
        vector <float> movie_rat = data->movie_user[id];
		sort(movie_rat.begin(), movie_rat.end());
		for (int s = 0; s < max_size; s ++) {*data_lib[max_size*j+s] = movie_rat[s]; cout << data_lib[max_size*j+s]<< " ";}
	}
	vector <float> movie_rat = data->movie_user[cur_movie_id];
	sort(movie_rat.begin(), movie_rat.end());
	for (int s = 0; s < max_size; s ++) {*data_lib[max_size*(user_list_movies.size()/3)+s] = movie_rat[s]; cout << data_lib[max_size*(user_list_movies.size()/3)+s]<< " ";}
	*max_data_size = max_size;
	cout <<"max_size " << max_size << " " << *max_data_size<< endl;
	cout <<"user_list_movies size " << user_list_movies.size()<<endl;
}

void knn(Data *data, float *cij_lib, int alpha, int k, string ifile, string ofile, bool probe)
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
    float e_out = 0;
        float * gpu_cij_lib;
        gpu_errchk(cudaMalloc((void **) &gpu_cij_lib, 17770 * 17770 * sizeof(float)));
        gpu_errchk(cudaMemcpy(gpu_cij_lib, cij_lib, 17770 * 17770 * sizeof(float),
                       cudaMemcpyHostToDevice));
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
		cout << "uid: "<<user_id << " movie_id: "<<cur_movie_id<<endl;
        vector<float> user_list_movies = data->user_movie[user_id];
        
        // This could be an array if it doesnt have pair.
        float * cij = new float[user_list_movies.size()/3];        
		cout << "i: " <<i<<" / "<<val_uid.size()<<endl;
        ////////////////////////////////////////////////////////
	    float *data_lib;

		int data_lib_size;		
		//organize(data, &data_lib, cur_movie_id, user_list_movies, &data_lib_size);
    	data_lib_size = data->movie_user[cur_movie_id].size();


    	for (int j = 0; j < user_list_movies.size()/3; j++)
    	{   
        	int movie_id = user_list_movies[3*j];
	        vector <float> movie_rat = data->movie_user[movie_id];
    	    data_lib_size = data_lib_size < movie_rat.size() ? movie_rat.size() : data_lib_size;
    	}   
    	data_lib = new float[  (user_list_movies.size()/3+1) * data_lib_size];//(float *) malloc(sizeof (float) * (user_list_movies.size()/3+1) * max_size);
    	cout << "emmm"<<endl;
    	memset(data_lib, -1, sizeof(float)*(user_list_movies.size()/3+1)*data_lib_size);
	    for (int j = 0; j < user_list_movies.size()/3; j++)
    	{   
        	int id = user_list_movies[3*j];
	        vector <float> movie_rat = data->movie_user[id];
        	for (int s = 0; s < movie_rat.size(); s ++) {data_lib[data_lib_size*j+s] = movie_rat[s];}
			for (int s = movie_rat.size(); s < data_lib_size; s ++) {data_lib[data_lib_size*j+s] = 0;}
   		}   
	    vector <float> movie_rat = data->movie_user[cur_movie_id];
    	//sort(movie_rat.begin(), movie_rat.end());
    	for (int s = 0; s < movie_rat.size(); s ++) {data_lib[data_lib_size*(user_list_movies.size()/3)+s] = movie_rat[s];}
        for (int s = movie_rat.size(); s < data_lib_size; s ++) {data_lib[data_lib_size*(user_list_movies.size()/3)+s] = 0;}

   		cout <<"user_list_movies size " << user_list_movies.size()<<endl;
		cout << "done organizing"<<endl;
		float * gpu_data_lib;

		cout << "hmm"<<endl;
		gpu_errchk(cudaMalloc((void **) &gpu_data_lib, (user_list_movies.size()/3 + 1)*data_lib_size * sizeof(float)));
		gpu_errchk(cudaMemcpy(gpu_data_lib, data_lib, (user_list_movies.size()/3 + 1)* data_lib_size * sizeof(float), cudaMemcpyHostToDevice));
		cout << "max index: "<<(user_list_movies.size()/3 + 1)* data_lib_size<<endl;	
        // Change into array.
        float *arr = new float[user_list_movies.size()];
        copy(user_list_movies.begin(), user_list_movies.end(), arr);
        
        float * gpu_user_list_movies;
        gpu_errchk(cudaMalloc((void **) &gpu_user_list_movies, user_list_movies.size() * sizeof(float)));
        gpu_errchk(cudaMemcpy(gpu_user_list_movies, arr, user_list_movies.size() * sizeof(float),
                   cudaMemcpyHostToDevice));
    
        float *gpu_out_cij;
        gpu_errchk(cudaMalloc((void **) &gpu_out_cij, user_list_movies.size()/3 * sizeof(float)));

		cout << "Calling kernal"<<endl;
        (cudaCallCij(2, 32, user_id, cur_movie_id, data_lib_size, gpu_data_lib, gpu_user_list_movies,
                                 gpu_out_cij, user_list_movies.size()/3, gpu_cij_lib, alpha));
        
        gpu_errchk(cudaMemcpy(cij, gpu_out_cij, user_list_movies.size()/3 * sizeof(float),
                   cudaMemcpyDeviceToHost));

        gpu_errchk(cudaMemcpy(cij_lib, gpu_cij_lib, 17770 * 17770 * sizeof(float), cudaMemcpyDeviceToHost));
		// FREEING
        cudaFree(gpu_out_cij);
        cudaFree(gpu_user_list_movies);
		cudaFree(gpu_data_lib);
		
		vector<pair<float, float>> cij_pair;
		for (int j = 0; j < user_list_movies.size()/3; j++)
        {
			cij_pair.push_back(make_pair(cij[j], cij[j]*user_list_movies[3*j+1]));
        }
		//######################################################
        ////////////////////////////////////////////////////////
        sort(cij_pair.begin(), cij_pair.end());

		delete [] arr;
		delete [] data_lib;
		delete [] cij;
        //######################################################
        float top = 0, bottom = 0;
        int loops = user_list_movies.size()/3 > k ? k : user_list_movies.size()/3;

        /////////////////////////////////////////////////////
        for (int j = 0; j < loops; j++)
        {
			cout << cij_pair[cij_pair.size() - 1 - j].first<< " ";
            top += cij_pair[cij_pair.size() - 1 - j].second;
            bottom += cij_pair[cij_pair.size() - 1 - j].first;
        }
        //######################################################
        float r_hat = bottom == 0? 3 : top/bottom;
        e_out += pow((val_ratings[i] - r_hat), 2);
		cout << "r_hat: "<<r_hat << " real: "<< val_ratings[i]<<endl;
    }
    e_out = pow(e_out / val_uid.size(), 0.5) ;
    cout << "error: "<<e_out<<endl;
    txtOut.close();
}

//This is where the execution of program begins
int main()
{
    Data data;
    
    //data.read_data("valid.txt", true);
    data.read_data("resource/small_train.txt", true);
    
    string ifile = "resource/small_probe.txt";
    string ofile = "resource/small_knn_probe.txt";
    
    
    float *cij_lib = new float[17770*17770];
    int alpha = 300;
    int k = 30;
    auto start = high_resolution_clock::now();
    
	cout << "LMAO"<<endl;
	memset(cij_lib, 0, sizeof(float)*17770*17770);
    knn(&data, cij_lib, alpha, k, ifile, ofile, true);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop-start);
    cout << "duration: "<< duration.count() <<endl;
    delete [] cij_lib;
    
    return 0;
}

