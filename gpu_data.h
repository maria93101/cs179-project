#include<vector>
#include<string>
#include<map>
#include<tuple>
using namespace std;

class Data
{
  private:
    // This prolly should be public too but we will see
    vector <vector<int>> all_data;
    //int max_user_id;
    //int max_movie_id;
  public:
    Data();
    // The other option is making a map of int, vectors of pairs. 
    // Alternatively, we can compaute it each time from all_data
    map<int, vector<float>> movie_user;
    map<int, vector<float>> user_movie;

    float * movie_user_list(int);
    int movie_user_list_num(int);
    /* 
     * Given an id (of user or movie), return the 
     * list of ratings from that movie or user. 
     * mu_orient = True for input = movie id, 
     * mu_orient = Fale for input being a user id
     */  
    vector<float> get_rats_list(int id, bool mu_orient);
    
    /*
     * Returns the average rating for movie or user
     * mu_orient = True for input - movie. 
     */
    float get_avg_rats(int id, bool mu_orient);
    
    /*
     * Reads in data at filename. Set mu = True if the 
     * file is mu formatted. False otherwise.
     */ 
    void read_data(string filename, bool mu);
    int max_user_id;
    int max_movie_id;
};
