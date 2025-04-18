#!/bin/bash

# 定义要删除的 screen 会话 ID 列表
session_ids=(
    2649012
    2638128
)

# 遍历并删除每个 screen 会话
for session_id in "${session_ids[@]}"; do
    screen -S "$session_id" -X quit
    echo "已删除 screen 会话 ID: $session_id"
done