# 计算机体系结构大作业

SYSU Computer Architecture Project

## Plan B: Parallel BFS on Many-core (GPU-CUDA)

使用 Nvidia GPU 的 CUDA 计数完成多核并行广度优先算法

### 实验环境

linux + cuda，需要在加载英伟达 GPU 的平台上，配置好 cuda，可以使用 `nvidia-smi` 来检测是否成功配置

### 编译

运行 `make.sh`，编译结果保存在 ./bin/bfs_test

若权限不足，执行 `chmod u+x make.sh`  

> 编译耗时比较久，请耐心等待

### 运行

运行 `./bin/bfs_test` 

若权限不足，执行 `chmod u+x run.sh`
