#!/bin/bash

# 创建 bin 目录（如果不存在）
if [ ! -d "bin" ]; then
    mkdir bin
fi

nvcc -std=c++11 src/main.cpp src/graph.cpp src/cuda_bfs.cu src/cpu_bfs.cpp src/utils.cpp -o ./bin/bfs_test

if [ $? -eq 0 ]; then
    echo "编译成功，可执行文件 bin/bfs_test"
else
    echo "编译失败"
fi
