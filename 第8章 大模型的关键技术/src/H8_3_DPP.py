
import numpy as np

def scatter_reduce(data, num_nodes):
    # 假设data是每个节点上的初始数组
    # num_nodes是工作节点数目
    
    # 第一步：每个节点把本设备上的数据分成N个区块
    local_blocks = np.array_split(data, num_nodes)
    
    # 第二步：在每个节点上累加其他节点一个块的数据
    for i in range(num_nodes):
        other_blocks = [local_blocks[j] for j in range(num_nodes) if j != i]
        local_blocks[i] += np.sum(other_blocks, axis=0)
    
    # 第三步：每个节点上都有一个包含局部最后结果的区块
    final_result = np.sum(local_blocks, axis=0)
    
    return final_result

# 示例
num_nodes = 3
data_size_per_node = 3
total_data_size = num_nodes * data_size_per_node

# 生成随机数据作为每个节点上的初始数组
data = np.random.randint(0, 10, total_data_size)

# 模拟Scatter-Reduce过程
result = scatter_reduce(data, num_nodes)

# 打印结果
print("初始数据：", data)
print("最终结果：", result)


def allgather(local_blocks, num_nodes):
    all_blocks = [np.empty_like(local_blocks) for _ in range(num_nodes)]
    
    for i in range(num_nodes):
        # 第一次迭代直接复制本地块到目标块
        all_blocks[i][:] = local_blocks[i]
    
    for _ in range(num_nodes - 1):
        # 迭代过程中交换块数据
        for i in range(num_nodes):
            target_node = (i + 1) % num_nodes
            # 发送当前节点的块到目标节点
            np.copyto(all_blocks[target_node], local_blocks[i])
            # 接收目标节点的块到当前节点
            np.copyto(local_blocks[i], all_blocks[target_node][i])
    
    return all_blocks

# 示例
num_nodes = 4
data_size_per_node = 5
total_data_size = num_nodes * data_size_per_node

# 生成随机数据作为每个节点的初始数组
local_data = np.random.randint(0, 10, (num_nodes, data_size_per_node))

# 模拟Allgather过程
result_blocks = allgather(local_data, num_nodes)

# 打印结果
print("初始数据块：", local_data)
print("Allgather结果块：", result_blocks)
