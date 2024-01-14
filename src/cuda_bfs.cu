#include "../include/cuda_bfs.cuh"

using namespace std;

#define DEBUG(x)
#define N_THREADS_PER_BLOCK (1 << 5)


// 初始化 cuda 中的数组
__global__
void init_cuda_array(int n, int *d_arr, int value, int start_index) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid == start_index) {
		d_arr[start_index] = 0;
	}
	else if (tid < n) {
		d_arr[tid] = value;
	}
}


// 已知图和当前队列，寻找下一层遍历的队列
__global__
void find_next_queue(int *adjacency_list, int *edge_offset, int *edges_size, int *distance,
		int queue_size, int *cur_queue, int *next_queue_size, int *next_queue, int level) {

	// 获取线程信息
	const int tid = blockIdx.x * blockDim.x + threadIdx.x; 
	// 并行遍历队列
	if (tid < queue_size) {
		int current = cur_queue[tid];
		for (int i = edge_offset[current]; i < edge_offset[current] + edges_size[current]; ++i) {
			int v = adjacency_list[i];
			if (distance[v] == INT_MAX) {
				distance[v] = level + 1;
				int position = atomicAdd(next_queue_size, 1);
				next_queue[position] = v;
			}
		}
	}
}


void cuda_bfs(int start, Graph &my_graph, vector<int> &distance, vector<bool> &is_visited) {


	const int N_BLOCKS = (my_graph.vertex_num + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;
	const int NEXT_QUEUE_SIZE = 0;

	// 初始化 cuda 中的变量
	int *device_adjacency_list;
	int *device_edges_offset;
	int *device_edges_size;
	int *device_first_queue;
	int *device_second_queue;
	int *device_next_queue_size;
	int *device_distance_array; // output

	// 初始化位于 cpu 中的变量
	int cur_queue_size = 1;
	int level = 0;

	// 在 gpu 上分配内存空间，并输入图数据
	int vertex_size = my_graph.vertex_num * sizeof(int);
	int adjacency_list_size = my_graph.adjacency_list.size() * sizeof(int);
	cudaMalloc((void **)&device_adjacency_list, adjacency_list_size);
	cudaMalloc((void **)&device_edges_offset, vertex_size);
	cudaMalloc((void **)&device_edges_size, vertex_size);
	cudaMalloc((void **)&device_first_queue, vertex_size);
	cudaMalloc((void **)&device_second_queue, vertex_size);
	cudaMalloc((void **)&device_distance_array, vertex_size);
	cudaMalloc((void **)&device_next_queue_size, sizeof(int));


	cudaMemcpy(device_adjacency_list, &my_graph.adjacency_list[0], adjacency_list_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_edges_offset, &my_graph.edge_offset[0], vertex_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_edges_size, &my_graph.edges_size[0], vertex_size, cudaMemcpyHostToDevice);
	cudaMemcpy(device_next_queue_size, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(device_first_queue, &start, sizeof(int), cudaMemcpyHostToDevice);

	clock_t start_time = clock();
	distance = vector<int> (my_graph.vertex_num, INT_MAX);
	distance[start] = 0;
	cudaMemcpy(device_distance_array, distance.data(), vertex_size, cudaMemcpyHostToDevice);

	while (cur_queue_size > 0) {
		int *device_cur_queue;
		int *device_next_queue;
		if (level % 2 == 0) {
			device_cur_queue = device_first_queue;
			device_next_queue = device_second_queue;
		}
		else {
			device_cur_queue = device_second_queue;
			device_next_queue = device_first_queue;
		}
		find_next_queue<<<N_BLOCKS, N_THREADS_PER_BLOCK>>> (device_adjacency_list, device_edges_offset, device_edges_size, device_distance_array,
				cur_queue_size, device_cur_queue, device_next_queue_size, device_next_queue, level);
		cudaDeviceSynchronize();
		cudaMemcpy(&cur_queue_size, device_next_queue_size, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(device_next_queue_size, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice);
		level++;
	}

	cudaMemcpy(&distance[0], device_distance_array, vertex_size, cudaMemcpyDeviceToHost);
	clock_t end_time = clock();
	double duration = ((double)(end_time - start_time));
	printf("Elapsed time for naive linear GPU implementation (without copying graph) : %.1lf ms.\n", duration);

	// 释放 gpu 内存空间
	cudaFree(device_adjacency_list);
	cudaFree(device_edges_offset);
	cudaFree(device_edges_size);
	cudaFree(device_first_queue);
	cudaFree(device_second_queue);
	cudaFree(device_distance_array);
}
