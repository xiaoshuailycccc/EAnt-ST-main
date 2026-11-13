import torch
import math

def generate_spatial_patches(alpha=1.0, beta=1.0, gamma=1.0, penalty_rho=0.1):
    print(">>> [Stage 2] 开始执行 ST-Ant 受限游走机制 (带软跳跃保护)...")
    
    # 加载第一阶段的先验知识
    data = torch.load('stage1_outputs.pt')
    tau = data['tau']                  
    dist = data['dist']                
    adj = data['adj']                  
    V = data['V']                      
    start_nodes = data['start_nodes']  
    num_nodes = data['num_nodes']
    k_ants = len(start_nodes)
    
    # 确保 Patch 长度足够覆盖所有节点 (加入少量冗余用于软惩罚的 Overlap)
    patch_length = math.ceil(num_nodes / k_ants) + 3 
    
    patches = [[node.item()] for node in start_nodes]
    M = torch.ones(num_nodes, dtype=torch.float32) 
    
    # 对起点进行软惩罚
    for node in start_nodes:
        M[node] *= penalty_rho

    # 预计算拉普拉斯空间相似度
    V_sq = torch.sum(V**2, dim=1, keepdim=True)
    topo_dist_sq = V_sq - 2 * torch.mm(V, V.t()) + V_sq.t()
    topo_sim = torch.exp(-topo_dist_sq)

    # 预计算距离倒数
    safe_dist = torch.clamp(dist, min=1e-2)
    inv_dist = 1.0 / safe_dist
    inv_dist.fill_diagonal_(0) 

    # 蚂蚁并行迭代寻路
    for step in range(1, patch_length):
        for ant_idx in range(k_ants):
            curr_node = patches[ant_idx][-1]
            
            # 1. 计算全局多维启发式概率分子 (包含距离、流量、拓扑和软惩罚)
            p_tau = torch.pow(tau[curr_node], alpha)
            p_topo = torch.pow(topo_sim[curr_node], beta)
            p_dist = torch.pow(inv_dist[curr_node], gamma)
            
            base_probs = p_tau * p_topo * p_dist * M
            
            # 2. 尝试严格物理拓扑约束 (只允许走邻居)
            strict_probs = base_probs * adj[curr_node]
            
            # 3. 改进的跃迁评估逻辑 (彻底消灭 1D-CNN 噪声源)
            if strict_probs.sum() < 1e-8:
                # 触发软邻居降级跳跃 (Soft Local Hop)
                # 放弃严格 adj 约束，但依然受 p_dist (物理距离倒数) 强烈制约
                # 确保蚂蚁即使跳跃，也是跳到物理上最近的未访问节点，而不是全局随机瞬移
                fallback_probs = base_probs.clone()
                fallback_probs[curr_node] = 0.0 # 禁止原地踏步
                
                if fallback_probs.sum() < 1e-8:
                    # 极端兜底：如果连 fallback 都找不到，只看软惩罚掩码，找一个最空闲的节点
                    next_node = torch.argmax(M).item()
                else:
                    # 在附近（距离优先）找一个未惩罚的节点跳跃
                    fallback_probs = fallback_probs / fallback_probs.sum()
                    next_node = torch.multinomial(fallback_probs, 1).item()
            else:
                # 正常归一化与采样 (有路可走)
                strict_probs = strict_probs / strict_probs.sum()
                next_node = torch.multinomial(strict_probs, 1).item()
            
            # 记录路径，并施加信息素软惩罚
            patches[ant_idx].append(next_node)
            M[next_node] *= penalty_rho 

    patches_tensor = torch.tensor(patches, dtype=torch.long)
    
    # ================= 打印与统计 =================
    print("\n" + "="*50)
    print(">>> ST-Ant 空间 Patch 划分结果明细")
    print("="*50)
    
    for i in range(k_ants):
        print(f"Patch {i:02d}: {patches_tensor[i].tolist()}")

    total_elements = patches_tensor.numel()
    unique_nodes = torch.unique(patches_tensor)
    total_unique = len(unique_nodes)
    coverage_rate = (total_unique / num_nodes) * 100
    overlap_count = total_elements - total_unique 

    print("\n" + "="*50)
    print(">>> 宏观统计信息 (用于论文 Experimental 章节)")
    print("="*50)
    print(f"| 路网总节点数 (N)       : {num_nodes}")
    print(f"| 生成 Patch 数量 (K)    : {k_ants}")
    print(f"| 单个 Patch 长度 (L)    : {patch_length}")
    print(f"| 实际覆盖的独立节点数   : {total_unique}")
    print(f"| 充当桥梁的 Overlap 节点: {overlap_count} (体现 Hub 特性)")
    print(f"| 全局节点覆盖率         : {coverage_rate:.2f}%")
    print("="*50)
    
    return patches_tensor

if __name__ == "__main__":
    # 执行寻路，返回 Patch 张量
    # 🌟 调整策略：流量相似度(alpha=2.5)主导，拓扑(beta=1.0)和物理距离(gamma=1.0)辅助
    final_patches = generate_spatial_patches(alpha=2.5, beta=1.0, gamma=1.0, penalty_rho=0.1)