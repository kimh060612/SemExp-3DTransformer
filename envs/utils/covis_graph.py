import cv2
import numpy as np
import torch
from scipy.spatial import cKDTree

class Vec:
    def __init__(self, id, vector):
        self.id = id
        self.vector = vector

    def __eq__(self, other):
        return self.id == other.id
    
    def __ne__(self, other):
        return self.id != other.id
    
    def __lt__(self, other):
        return self.id < other.id
    
    def __le__(self, other):
        return self.id <= other.id
    
    def __gt__(self, other):
        return self.id > other.id
    
    def __ge__(self, other):
        return self.id >= other.id
    
    def __hash__(self):
        return hash(self.id)
    
class CoVisibilityGraph:
    T = 0.5
    
    def __init__(self, num_steps, device):
        self.vertex = []
        self.step_size = num_steps
        self.now = 0
        self.adj_matrix = np.ones((num_steps + 1, num_steps + 1)) # attention mask로 바꿀 예정
        self.adj_matrix = np.triu(self.adj_matrix, k=1)
        self.device = device
        
    def _vec_sim(self, vector1: torch.Tensor, vector2: torch.Tensor):
        dot = torch.dot(vector1, vector2) # 제대로 작동
        norm1 = torch.norm(vector1, p=2)
        norm2 = torch.norm(vector2, p=2)
        similarity = (dot / (norm1 * norm2)).item()
        return (1 + similarity) / 2 # 0~1까지의 값으로 normalize
    
    def add_node(self, vector: torch.Tensor):
        edge = np.array([ False for _ in range(self.step_size + 1) ])
        for idx, (t, v) in enumerate(self.vertex):
            if self._vec_sim(v, vector) > self.T:
                edge[idx] = True
        self.vertex.append((self.now, vector))
        if self.now + 1 >= self.step_size:
            del self.vertex[0]
        self.now = (self.now + 1) % self.step_size
        self.adj_matrix[:,self.now] = edge.T
    
    def get_attn(self):
        return torch.from_numpy(self.adj_matrix).to(self.device)

# vector matching
def match_vector(vector1, vector2):
    '''
    Execute Cosine Similarity
    '''
    dot = torch.dot(vector1, vector2) # 제대로 작동
    norm1 = torch.norm(vector1, p=2)
    norm2 = torch.norm(vector2, p=2)
    similarity = (dot / (norm1 * norm2)).item()
    return (1+similarity)/2 # 0~1까지의 값으로 normalize

def build_co_visibility_graph(vector_id:list, vector_list:list):
    co_visibility_graph = CoVisibilityGraph()
    for i in range(len(vector_id)):
        for j in range(i + 1, len(vector_id)):
            vector1 = Vec(vector_id[i] ,vector_list[i]) # id: vector_id[i]
            vector2 = Vec(vector_id[j], vector_list[j]) # id: vector_id[j]
            similarity = match_vector(vector1.vector, vector2.vector)
            co_visibility_graph.add_edge(vector1, vector2, similarity)
            #print("Vec:", vector1.id, vector2.id, "Sim:", similarity)
    return co_visibility_graph

def kruskal_mst(co_visibility_graph):
    parent = {} # Key: Vec, value: Vec
    rank = {} # Key: Vec, value: rank

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]
    
    def union(node1, node2):
        root1 = find(node1) # node 반환 - node: Vec obj
        root2 = find(node2) # node 반환 - node: Vec obj
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
                if rank[root1] == rank[root2]:
                    rank[root2] += 1
    
    edges = co_visibility_graph.get_edges() # ok
    sorted_edges = sorted(edges.items(), key = lambda item: item[1], reverse=True) # [(key, value), (key, value)] <= 내림차순    

    for node in co_visibility_graph.graph: # 초기화
        parent[node] = node # node 개수
        rank[node] = 0

    mst = CoVisibilityGraph()

    for edge in sorted_edges:
        (vector1, vector2), similarity = edge
        if find(vector1) != find(vector2):
            union(vector1, vector2)
            mst.add_edge(vector1, vector2, similarity)
    return mst


def add_image_to_mst(mst, new_vector_id, new_vec:torch.Tensor):
    new_vector = Vec(new_vector_id, new_vec)

    best_edges = {}
    for existing_vector in mst.get_nodes():
        similarity = match_vector(existing_vector.vector, new_vector.vector)
        #print("Edges:", existing_vector.id, new_vector.id, "Sim:", similarity)
        best_edges[(new_vector, existing_vector)] = similarity
    
    sorted_best_edges = sorted(best_edges.items(), key = lambda item: item[1], reverse=True)

    parent = {}
    rank = {}
    for node in mst.get_nodes() + [new_vector]:
        parent[node] = node
        rank[node] = 0

    def find(node):
        if parent[node] != node:
            parent[node] = find(parent[node])
        return parent[node]

    def union(node1, node2):
        root1 = find(node1)
        root2 = find(node2)
        if root1 != root2:
            if rank[root1] > rank[root2]:
                parent[root2] = root1
            else:
                parent[root1] = root2
                if rank[root1] == rank[root2]:
                    rank[root2] += 1

    for edge in sorted_best_edges:
        (vector1, vector2), similarity = edge
        if find(vector1) != find(vector2):
            union(vector1, vector2)
            mst.add_edge(vector1, vector2, similarity)
            break
