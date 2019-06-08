Motivation


Matrix factorization and K-nearest neighbors have copious uses such as in recommender systems, data compression, pagerank, and other machine learning applications. As the input data set is often large, matrix factorization solutions are often time-consuming. KNN, too involves computing similar operations for each of the expected values. Implementing these algorithms on the GPU can speed up this process and allow these algorithms to be used more easily and on larger datasets. For our CS 179 project, we implemented various recommendation systems using the GPU. 


High-level overview of algorithms


We implemented a matrix factorization algorithm using stochastic gradient descent. During each iteration of SGD, we draw a random example from the training data and perform a gradient-descent-like update with respect to the single example. We ran this algorithm on a small subset of data from the Netflix challenge, in the format “user id, movie id, date, rating.”


Each item can be represented by a vector qi and each user can be represented by a vector pu such that the dot product of the two vectors is the expected rating.


The qi and pu vectors can be found in such a way that the square error difference between their dot product and the known rating in the user-item matrix is minimized.


In order to make our model more generalizable and reduce overfitting, we add a regularization factor to serve as a penalty term.


We used knn with pearson’s correlation to calculate the correlation (cij) between two movies and sort based on this correlation, and finally use the highest correlations for the calculating r_hat. Any number with a bar over it is the mean. The two movie rating lists are x_i and x_j. L is the size of the lists. 






For a given movie id and uid, we get a list of movies that the uid has watched. For each other_movie_id in the list of movies, we look and see the ratings that both users have rating on the two movies. We calculated pij, and make a list of them that is summed over. Nij is total number of the common ratings by two users. This gives us cij = pij * nij / (pij + alpha). 
R_hat = sum of cij * rating for that movie / sum of cij
 


The sort and addition were both completed.  


GPU optimizations and specifics 


Matrix Factorization:
To convert from CPU code to GPU code, we converted all the C++ style vectors to C style arrays. We also removed all usage of the Eigen linear algebra library. This change may have actually made our code slower, as Eigen is highly optimized for linear algebra operations. In addition we were using Sparse Matrices to only iterate through the non zero elements of our matrix instead of all NUM_USER X NUM_MOVIES, which is not possible with the GPU implementation. We replaced the Eigen commands with CUBLAS linear algebra operations which should result in optimizations.


We had a lot of implementation issues with the svd. Currently, the code compiles but encounters a runtime error on the first cublas call (line 122). The value of the vectors being dotted together with cublasDdot are correct, the arrays seem to be stored correctly in device memory, and the provided error checking functions in helper_cuda.cpp didn’t catch any errors, so we concluded that the issue probably lies in combining device and host code.


KNN:
The original idea was to parallel the calculation for a single prediction value at a time. This would mean 3 parts to parallelize: the part to get all the correlations between movies that the user has watched and the current movie being predicted, the sorting, and the adding of the correlations. However, several difficulties were encountered. The original setup involved using many standard c++ library that were not standard in cuda. This meant that things like vectors and pairs were hard to use. This resulted in a lot of duplicated code to get certain sizes and to create certain arrays. In the end, we could not get this part to work due to memory issues. Another issue that came up was our data reading object. To make getting data easily, we had created an object which maps a user id or movie to a list of corresponding movie or user information. To combat this, we created a 1D array such that it is max_length * (num_movies + 1), where max_length was the largest lengthed vector for the list of movies (returned by our object). Other numerous complications involved the amount of memory required. Because we have a different set of list for each pair of movie ids, this resulted in a lot of malloc calls and very quickly the code fails due to malloc failure. Attempting to shift around and make less malloc calls complicated the code too much. 


The idea behind the merge sort is to use each thread to sort individual pieces of the array. We start with sorting only the two nearby values for the entire list, and then double that each time until everything is sorted. 


The addition is very similar to what was used in set 5. We used reduction to sum over everything in the array. 


There are still some bugs from the mergesort and addition.


Code structure (description of where important files are)


Matrix Factorization:
The majority of the cuda and C++ code is in svd.cu. The header file is svd.cuh. The main method that calls the setup and model training functions is in svd_GPU.cpp.


KNN:
Most of the code is in merge_knn.cpp. The Cuda code for sorting and addition is in knn.cu. The failed code described above is at parallel_knn.cu and cuda_knn.cpp


Instructions on how to run smaller components of the project (In addition to demo script) 


To run the matrix multiplication, use “make clean svd” to compile and run ./svd.


To run knn, use “make clean knn” to compile and run ./knn


Provide code output when run on Titan, especially if it takes a while to run


KNN GPU:
Processing finished
error: 1.23937
duration: 136468139


KNN CPU:
Processing finished
error: 1.10155
duration: 657224031


SVD CPU:
loading data took: 10446350
training model took: 3337606433 (35 epochs)
predicting valid took: 305761


We don’t have code output for the GPU version of SVD since it encounters runtime errors :(