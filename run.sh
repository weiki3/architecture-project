#!/bin/bash

echo "使用小型图测试"
./bin/bfs_test < ./graph_data/small_graph.txt
echo "使用大型图测试"
./bin/bfs_test < ./graph_data/big_graph.txt
