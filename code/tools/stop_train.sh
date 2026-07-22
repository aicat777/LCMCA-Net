#!/bin/bash

# 查找相关进程
echo "查找 torch.distributed.launch 相关进程："
echo ""

# 使用数组存储进程信息
mapfile -t processes < <(ps aux | grep -E "torch\.distributed\.launch" | grep -v grep)

if [ ${#processes[@]} -eq 0 ]; then
    echo "未找到相关进程。"
    exit 0
fi

# 显示进程
echo "编号 | PID   | 命令"
echo "----|-------|----------------"
for i in "${!processes[@]}"; do
    pid=$(echo "${processes[$i]}" | awk '{print $2}')
    cmd=$(echo "${processes[$i]}" | awk '{for(i=11;i<=NF;i++) printf $i" "; print ""}')
    printf "%-4d| %-6s| %s\n" "$i" "$pid" "$cmd"
done

echo ""
read -p "输入要终止的进程编号（多个用空格分隔，或输入 'all'）： " selection

if [ "$selection" = "all" ]; then
    echo "终止所有进程..."
    pkill -f "torch.distributed.launch"
    echo "完成。"
else
    for num in $selection; do
        if [[ "$num" =~ ^[0-9]+$ ]] && [ "$num" -lt "${#processes[@]}" ]; then
            pid=$(echo "${processes[$num]}" | awk '{print $2}')
            echo "终止进程 PID: $pid"
            kill $pid
        fi
    done
    echo "完成。"
fi