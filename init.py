import numpy as np
import pickle
import torch
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import os

def build_prior_knowledge(data_path, embed_dim=8, k_ants=None):
    print(">>> [Stage 1] 正在为 Ant-ST 模型加载并处理 PEMS 数据...")
    
    # 1. 加载流量数据
    dataset = np.load(data_path)
    data = dataset['data']  
    num_nodes = data.shape[1]
    flow_data = data[:, :, 0].T 
    
    # 2. 计算流量语义相似度 (初始信息素 tau)
    # 仅在训练切分(前 70%)上计算, 避免验证/测试集信息泄漏到离线先验中
    print(">>> [Stage 1] 计算流量皮尔逊相关系数 (仅训练切分, 含 NaN 鲁棒处理)...")
    split = int(flow_data.shape[1] * 0.7)
    corr_matrix = np.corrcoef(flow_data[:, :split])
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    tau = (corr_matrix - corr_matrix.min()) / (corr_matrix.max() - corr_matrix.min())
    tau_tensor = torch.tensor(tau, dtype=torch.float32)

    # 3. 读取 PKL 文件构建物理距离与邻接矩阵
    base_dir = os.path.dirname(data_path)
    stem = os.path.splitext(os.path.basename(data_path))[0]
    pkl_file = os.path.join(base_dir, f'{stem}.pkl')
    
    dist_matrix = np.full((num_nodes, num_nodes), np.inf)
    np.fill_diagonal(dist_matrix, 0)
    
    if os.path.exists(pkl_file):
        print(f">>> [Stage 1] 读取真实路网 PKL 文件: {pkl_file}")
        try:
            with open(pkl_file, 'rb') as f:
                # 兼容 python 2/3 的 pickle 序列化差异
                try:
                    pkl_data = pickle.load(f)
                except UnicodeDecodeError:
                    pkl_data = pickle.load(f, encoding='latin1')
                
                # 提取 adj_mx (通常在元组的第三个位置)
                if isinstance(pkl_data, tuple) and len(pkl_data) >= 3:
                    adj_mx = pkl_data[2]
                else:
                    adj_mx = pkl_data
            
            # 智能判断 PKL 存的是距离还是权重
            if np.max(adj_mx) > 2.0:
                # 存的是真实距离 (0表示不通或自身)
                print(">>> 检测到 PKL 存储的是真实距离矩阵。")
                dist_matrix = adj_mx.copy()
                dist_matrix[dist_matrix == 0] = np.inf
                np.fill_diagonal(dist_matrix, 0)
            else:
                # 存的是高斯权重 (0表示不通，(0,1]表示连通)
                print(">>> 检测到 PKL 存储的是归一化权重矩阵，正在反演出距离矩阵...")
                valid_mask = adj_mx > 1e-5
                # 将权重取倒数作为虚拟距离，权重越大，虚拟距离越小，符合转移概率公式
                dist_matrix[valid_mask] = 1.0 / adj_mx[valid_mask]
                np.fill_diagonal(dist_matrix, 0)
                
        except Exception as e:
            print(f"读取 PKL 出错: {e}，将使用流量反演图。")
    else:
        print(f">>> [Stage 1] 未找到 PKL，构建全连接模拟图...")
        dist_matrix = np.exp(-corr_matrix)
        np.fill_diagonal(dist_matrix, 0)
    
    # 强制对称化，满足谱图理论
    dist_matrix = np.minimum(dist_matrix, dist_matrix.T)

    # 严格拓扑约束：非无穷大即为有边相连
    A_binary = (dist_matrix != np.inf).astype(np.float32)
    np.fill_diagonal(A_binary, 0)
    adj_tensor = torch.tensor(A_binary, dtype=torch.float32)

    # 将无穷大替换为极大值
    max_valid_dist = dist_matrix[dist_matrix != np.inf].max() if np.any(dist_matrix != np.inf) else 1.0
    dist_matrix[dist_matrix == np.inf] = max_valid_dist * 10
    dist_tensor = torch.tensor(dist_matrix, dtype=torch.float32)

    # 4. 计算拉普拉斯谱空间嵌入
    print(">>> [Stage 1] 计算标准化拉普拉斯矩阵...")
    valid_dists = dist_matrix[A_binary == 1]
    sigma = np.std(valid_dists) if len(valid_dists) > 0 else 1.0
    W = np.exp(-(dist_matrix ** 2) / (sigma ** 2)) * A_binary 
    
    D = np.diag(np.sum(W, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
    L_sys = np.eye(num_nodes) - D_inv_sqrt @ W @ D_inv_sqrt
    
    eigenvalues, eigenvectors = np.linalg.eigh(L_sys)
    V_embeddings = eigenvectors[:, 1:embed_dim+1]
    V_tensor = torch.tensor(V_embeddings, dtype=torch.float32)

    # 5. K-Means 引导播种
    # 蚂蚁数 K 以论文实验设置为准: PEMS04=16, PEMS07=30, PEMS08=13, NYCTaxi=10, NYCBike=10
    if k_ants is None:
        k_ants = {307: 16, 883: 30, 170: 13, 250: 10, 266: 10}.get(num_nodes, int(np.sqrt(num_nodes)))
    print(f">>> [Stage 1] 执行 K-Means 聚类，精准投放 {k_ants} 只蚂蚁...")
    kmeans = KMeans(n_clusters=k_ants, random_state=42, n_init=10).fit(V_embeddings)
    centroids = kmeans.cluster_centers_
    distances = cdist(centroids, V_embeddings)
    start_nodes = np.argmin(distances, axis=1)
    
    torch.save({
        'tau': tau_tensor,
        'dist': dist_tensor,
        'adj': adj_tensor,  
        'V': V_tensor,
        'start_nodes': torch.tensor(start_nodes, dtype=torch.long),
        'num_nodes': num_nodes
    }, 'stage1_outputs.pt')
    
    print(">>> [Stage 1] 完成！先验知识已保存。\n")

if __name__ == "__main__":
    build_prior_knowledge("data/PEMS04/PEMS04.npz", embed_dim=8, k_ants=16)