"""Classes for SimGNN modules."""

import torch

class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3,
                                                             self.args.filters_3))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN  =>  [num_nodes, filters_3]
        :return representation: A graph level representation vector.
        """
        # global_context： 所有节点的embedding求平均
        # torch.matmul(embedding, self.weight_matrix) shape =  [num_nodes, filters_3] * [filters_3, filters_3] = [num_nodes, filters_3]
        # dim = 0 表示 列求平均 
        # 所以 global_context 的shape 为 => [1, filters_3]  => [1, embedding_size] => [1, 32]
        # 也就是说， mean求得的是 所有节点的embedding表示的平均值
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        # transformed_global 就是 paper的 "C" , [1, embedding_size] => [1, 32]
        transformed_global = torch.tanh(global_context)
        # sigmoid_scores 的shape 为 [num_nodes, filters_3] * [filters_3, 1] = [num_nodes, 1]
        sigmoid_scores = torch.sigmoid(torch.mm(embedding, transformed_global.view(-1, 1)))
        
        # 为图的每一个节点都计算一个 att_embedding表示
        # representation.shape = [filters_3, num_nodes] * [num_nodes, 1] = [filters_3, 1]  =>  [32, 1] 
        representation = torch.mm(torch.t(embedding), sigmoid_scores)

class TenorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(TenorNetworkModule, self).__init__()
        self.args = args
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        # [filters_3 , filters_3 , tensor_neurons]  =>  [32 , 32 , 16] 
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.args.filters_3,
                                                             self.args.filters_3,
                                                             self.args.tensor_neurons))
        # [tensor_neurons , (2 * filters_3)]  =>  [16, 64]
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,
                                                                   2*self.args.filters_3))
        #   [tensor_neurons , 1]  => [16 , 1]                                              
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.
        :param embedding_2: Result of the 2nd embedding after attention.
        即： 传入的是两个图的 att_embedding  shape: [filters_3, 1]  =>  [32, 1]
        :return scores: A similarity score vector.
        """

        # calculate : scoring =  embedding_1 @ self.weight_matrix @ embedding_2
        # shape change:  T([32, 1]) * [32, 32*16]  =>  [1, 32] * [32, 32*16]  =>  [1, 32*16]  =>  [1, 512]
        scoring = torch.mm(torch.t(embedding_1), self.weight_matrix.view(self.args.filters_3, -1))

        # [1, 512]  =>  [32, 16]
        scoring = scoring.view(self.args.filters_3, self.args.tensor_neurons)
        
        # [16, 32] * [32, 1]  =>  [16, 1]
        scoring = torch.mm(torch.t(scoring), embedding_2)
        
        # calculate: block_scoring =  embedding_1_2  @  self.weight_matrix_block
        # [64, 1]
        combined_representation = torch.cat((embedding_1, embedding_2))
        
        # [16, 64] * [64, 1]  =>  [16, 1] 
        block_scoring = torch.mm(self.weight_matrix_block, combined_representation)
        
        # scoring + block_scoring  
        # 对应值相加
        # [16, 1] + [16, 1] = [16, 1]
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores
