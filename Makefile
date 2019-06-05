CXX = g++
CXXFLAGS = -std=c++0x -Wall -pedantic-errors -mcmodel=large -g

SRCS = cuda_knn.cpp gpu_data.cpp
OBJS = ${SRCS:.cpp=.o}

MAIN = knn

all: ${MAIN}

svd:
	${CXX} ${CXXFLAGS} svd.cpp -o svd

${MAIN}: 
	${CXX} ${CXXFLAGS} ${SRCS} -o ${MAIN}

.cpp.o:
	${CXX} ${CXXFLAGS} -c $< -o $@
clean:
	${RM} ${MAIN} svd ${OBJS} *.o *~.
