#include "../include/cuda_bfs.cuh"

using namespace std;

#define DEBUG(x)
#define N_THREADS_PER_BLOCK (1 << 5)



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


/*
 * Given a graph and a current queue computes next vertices (vertex frontiers) to traverse.
 */
__global__
void find_next_queue(int *adjacency_list, int *edge_offset, int *edges_size, int *distance,
		int queue_size, int *cur_queue, int *next_queue_size, int *next_queue, int level) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;  // thread id
	if (tid < queue_size) {  // visit all vertexes in a queue in parallel
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


void cuda_bfs(int start, Graph &G, vector<int> &distance, vector<bool> &is_visited) {


	const int n_blocks = (G.vertex_num + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

	// Initialization of GPU variables
	int *d_adjacencyList;
	int *d_edgesOffset;
	int *d_edgesSize;
	int *d_firstQueue;
	int *d_secondQueue;
	int *d_nextQueueSize;
	int *d_distance; // output


	// Initialization of CPU variables
	int currentQueueSize = 1;
	const int NEXT_QUEUE_SIZE = 0;
	int level = 0;

	// Allocation on device
	const int size = G.vertex_num * sizeof(int);
	const int adjacencySize = G.adjacency_list.size() * sizeof(int);
	cudaMalloc((void **)&d_adjacencyList, adjacencySize);
	cudaMalloc((void **)&d_edgesOffset, size);
	cudaMalloc((void **)&d_edgesSize, size);
	cudaMalloc((void **)&d_firstQueue, size);
	cudaMalloc((void **)&d_secondQueue, size);
	cudaMalloc((void **)&d_distance, size);
	cudaMalloc((void **)&d_nextQueueSize, sizeof(int));


	// Copy inputs to device

	cudaMemcpy(d_adjacencyList, &G.adjacency_list[0], adjacencySize, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edgesOffset, &G.edge_offset[0], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_edgesSize, &G.edges_size[0], size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_nextQueueSize, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_firstQueue, &start, sizeof(int), cudaMemcpyHostToDevice);
//	init_cuda_array<<<n_blocks, N_THREADS_PER_BLOCK>>> (G.vertex_num, d_distance, INT_MAX, start); // FOR SOME REASON USING THIS KERNEL DOESNT WORK
//	cudaDeviceSynchronize();

	auto start_time = chrono::steady_clock::now();
	distance = vector<int> (G.vertex_num, INT_MAX);
	distance[start] = 0;
	cudaMemcpy(d_distance, distance.data(), size, cudaMemcpyHostToDevice);

	while (currentQueueSize > 0) {
		int *d_currentQueue;
		int *d_nextQueue;
		if (level % 2 == 0) {
			d_currentQueue = d_firstQueue;
			d_nextQueue = d_secondQueue;
		}
		else {
			d_currentQueue = d_secondQueue;
			d_nextQueue = d_firstQueue;
		}
		find_next_queue<<<n_blocks, N_THREADS_PER_BLOCK>>> (d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
				currentQueueSize, d_currentQueue, d_nextQueueSize, d_nextQueue, level);
		cudaDeviceSynchronize();
		++level;
		cudaMemcpy(&currentQueueSize, d_nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(d_nextQueueSize, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice);
	}

	cudaMemcpy(&distance[0], d_distance, size, cudaMemcpyDeviceToHost);
	auto end_time = std::chrono::steady_clock::now();
	auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();
	printf("Elapsed time for naive linear GPU implementation (without copying graph) : %li ms.\n", duration);

	// Cleanup
	cudaFree(d_adjacencyList);
	cudaFree(d_edgesOffset);
	cudaFree(d_edgesSize);
	cudaFree(d_firstQueue);
	cudaFree(d_secondQueue);
	cudaFree(d_distance);
}
