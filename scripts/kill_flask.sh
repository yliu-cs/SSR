#!/bin/bash

apt install lsof

# 获取网络连接信息
lsof_result=$(lsof -i)

# 定义一个函数，用于杀死指定端口占用的进程
kill_process_by_port() {
    local port=$1
    local pids=$(echo "$lsof_result" | grep ":$port" | awk '{print $2}' | sort -u)

    if [ -n "$pids" ]; then
        for pid in $pids; do
            # 可以在这里添加进一步确认进程的逻辑
            # 例如查看进程的命令行参数等
            echo "Killing process $pid using port $port"
            kill $pid
        done
    else
        echo "No process found using port $port"
    fi
}

# 杀死占用端口 23456 的进程
kill_process_by_port 23456