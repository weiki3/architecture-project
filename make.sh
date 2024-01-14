nvcc -std=c++11 src/main.cpp src/graph.cpp src/cuda_bfs.cu src/cpu_bfs.cpp src/utils.cpp -o ./bin/bfs_test
&& echo "成功完成 cuda 编译"