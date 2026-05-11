import numpy as np
import torch
from torch_geometric.data import Data

class ArchitectureToGraphEncoder:
    def __init__(self,node_feature_dim=18, 
                 interpol_options=[8,10,12], 
                 kernel_options= [3,5,7],
                 expansion_options = [1,2,3,4,6], 
                 num_of_blocks = 5, max_depth = 4, 
                 min_res=192, max_res=256):
        self._node_feature_dim = node_feature_dim
        self._interpol_options = interpol_options
        self._kernel_options = kernel_options
        self._expansion_options = expansion_options
        self._num_of_blocks = num_of_blocks
        self._max_depth = max_depth
        self._min_resolution = min_res
        self._max_resolution = max_res

    def max_sequence_length(self):
        """Return the maximum number of tokens including the [CLS] token."""
        max_backbone_nodes = self._num_of_blocks * self._max_depth
        max_exit_nodes = max(0, self._num_of_blocks - 1) * 6
        return 1 + max_backbone_nodes + max_exit_nodes

    def build_graph_dataset(self, data, targets=None):
        if isinstance(data, dict):
            data = [data]

        if targets is not None:
            y_np = np.asarray(targets, dtype=np.float32)
            if y_np.ndim == 1:
                y_np = y_np.reshape(-1, 1)
            assert len(data) == y_np.shape[0], "Inputs and targets must have the same number of samples"

        graph_data = []
        input_resolutions= []
        for idx, arch in enumerate(data):
            if not isinstance(arch, dict):
                raise TypeError("GIN expects architectures as dictionaries. Skip encoding to integer strings for this surrogate.")

            x, edge_index = self._convert_architecture_to_graph(arch)
            graph = Data(
                x=torch.tensor(x, dtype=torch.float32),
                edge_index=torch.tensor(edge_index, dtype=torch.long),
            )

            if targets is not None:
                graph.y = torch.tensor(y_np[idx], dtype=torch.float32)

            graph_data.append(graph)
            normalized_resolution = self._normalize_resolution(arch['r'])
            input_resolutions.append(normalized_resolution)

        return graph_data, input_resolutions

    def build_sequence_dataset(self, data, targets=None):
        if isinstance(data, dict):
            data = [data]

        if targets is not None:
            y_np = np.asarray(targets, dtype=np.float32)
            if y_np.ndim == 1:
                y_np = y_np.reshape(-1, 1)
            assert len(data) == y_np.shape[0], "Inputs and targets must have the same number of samples"

        max_tokens = self.max_sequence_length() - 1
        n_samples = len(data)

        sequences = np.zeros((n_samples, max_tokens, self._node_feature_dim), dtype=np.float32)
        padding_masks = np.ones((n_samples, max_tokens + 1), dtype=bool)
        padding_masks[:, 0] = False
        input_resolutions = np.zeros((n_samples, 1), dtype=np.float32)
        targets_tensor = None

        if targets is not None:
            targets_tensor = torch.tensor(y_np, dtype=torch.float32)

        for idx, arch in enumerate(data):
            if not isinstance(arch, dict):
                raise TypeError("Transformer expects architectures as dictionaries.")

            x, _ = self._convert_architecture_to_graph(arch)
            seq_len = x.shape[0]
            if seq_len > max_tokens:
                raise ValueError(
                    f"Architecture with {seq_len} nodes exceeds transformer capacity of {max_tokens} nodes."
                )

            sequences[idx, :seq_len, :] = x
            padding_masks[idx, 1:seq_len + 1] = False
            input_resolutions[idx, 0] = self._normalize_resolution(arch['r'])

        return (
            torch.tensor(sequences, dtype=torch.float32),
            torch.tensor(input_resolutions, dtype=torch.float32),
            torch.tensor(padding_masks, dtype=torch.bool),
            targets_tensor,
        )
    
    def _convert_architecture_to_graph(self, arch):
        """
        Convertes encoded architecture from the sample space to a graph encoding 
        to be used for the GNN.

        Params:

            arch: dictionary with structure 
            {'ks': [...],         # kernel sizes\n
            'e': [...],          # expansion ratios\n
            'd': [...],          # depths\n
            't': [...],          # thresholds (for early exit)\n
            'r': int,            # resolution\n
            'e_ks': [...]        # exit layer configurations}\n
            and e_ks: \n
            [\n
            [[mod1_exit1], [mod2_exit1]],  # Exit after block 1\n
            [[mod1_exit2], [mod2_exit2]],  # Exit after block 2\n
            [[mod1_exit3], [mod2_exit3]],  # Exit after block 3\n
            [[mod1_exit4], [mod2_exit4]],  # Exit after block 4\n
            ] with mod = [ext, kern, exp, pool]
        Returns:
            x, edge_index: x is the node representations, shape [num_nodes, num_node_features=18]. 
            edge_index is the adjacency list, shape [2, num_edges]
        """
        x, edge_index_transposed = [], [] #for the result
        assert len(arch['ks']) == len(arch['e']) == sum(arch['d'])
        assert len(arch['d']) == self._num_of_blocks, f"{arch['d']} does not have length {self._num_of_blocks} as expected"
        assert len(arch["e_ks"]) ==  self._num_of_blocks - 1
        #setup
        node_index = 0  # global index over all nodes added to x
        conv_index = 0  # index over backbone convolutions (for arch['ks'] / arch['e'])
        #access the index of the last node of a backbone
        backbone_end_node_index = {block: -1 for block in range(self._num_of_blocks)}#used to connect the early exits to 

        for block in range(self._num_of_blocks):
            #CREATE BACKBONE NODES
            for depth in range(arch['d'][block]):
                conv_node = self.__create_convolution_node(block,
                                                           depth,
                                                           arch['ks'][conv_index],
                                                           arch['e'][conv_index],
                                                           is_exit=False)
                x.append(conv_node)
                edge_added=False
                if node_index == 0:
                    edge_added = True
                if depth !=0:
                    assert node_index != 0
                    edge_index_transposed.append([node_index-1, node_index])
                    edge_added = True

                if depth == arch['d'][block]-1: #if last node of the block
                    backbone_end_node_index[block] = node_index #record index

                if depth == 0 and block !=0: #if fist node of a block that isnt the first block
                    assert node_index != 0 #just in case
                    edge_index_transposed.append([backbone_end_node_index[block-1], node_index])
                    edge_added=True
                assert edge_added == True, "Something went wrong and a disconnected node was created"
                
                node_index +=1
                conv_index +=1

            #now create nodes for all early exits but skip last block as the last block cant have an early exit
            if block==self._num_of_blocks - 1:
                break
            #assertions
            assert len(arch["e_ks"][block]) == 2
            for layer_idx in range(2):
                assert len(arch["e_ks"][block][layer_idx]) == 4
            #end of assertions
            layer_1 = arch['e_ks'][block][0]
            layer_2 = arch['e_ks'][block][1]
            if arch['t'][block] != 1: #there is an exit at this block
                interpol_node_1 = self.__create_interpolation_node(block, 0,layer_1[0], arch['t'][block])
                x.append(interpol_node_1)
                if backbone_end_node_index[block] == -1:
                    raise Exception("bruh this shouldnt be -1")
                edge_index_transposed.append([backbone_end_node_index[block], node_index]) 
                node_index+=1

                conv_node_1 = self.__create_convolution_node(block, 0, layer_1[1], layer_1[2], is_exit=True)
                x.append(conv_node_1)
                edge_index_transposed.append([node_index-1, node_index])
                node_index+=1
                if layer_1[3]==1: #max pooling is present
                    pool_node_1 = self.__create_pooling_node(block, 0, arch['t'][block])
                    x.append(pool_node_1)
                    edge_index_transposed.append([node_index-1, node_index])
                    node_index+=1

                #now on to second layer
                if layer_2[0]!=0:#if it exists
                    interpol_node_2 = self.__create_interpolation_node(block, 1,layer_2[0], arch['t'][block])
                    x.append(interpol_node_2)
                    edge_index_transposed.append([node_index-1, node_index]) 
                    node_index+=1

                    conv_node_2 = self.__create_convolution_node(block, 1, layer_2[1], layer_2[2], is_exit=True)
                    x.append(conv_node_2)
                    edge_index_transposed.append([node_index-1, node_index])
                    node_index+=1
                    if layer_2[3]==1: #max pooling is present
                        pool_node_2 = self.__create_pooling_node(block, 1, arch['t'][block])
                        x.append(pool_node_2)
                        edge_index_transposed.append([node_index-1, node_index])
                        node_index+=1

        #convert to numpy and transpose edge_index, making it shape (2,Num of edges)
        x = np.asarray(x, dtype=np.float32)
        if len(edge_index_transposed) == 0:
            edge_index = np.empty((2, 0), dtype=np.int64)
        else:
            edge_index = np.asarray(edge_index_transposed, dtype=np.int64).T
        return x, edge_index

    def _normalize_resolution(self, resolution):
        denom = self._max_resolution - self._min_resolution
        if denom == 0:
            return 0.0
        return (resolution - self._min_resolution) / denom
    
    def __create_interpolation_node(self, block, layer, interpol_size, threshold):
        """Assumes block and layers start counting from 0"""
        node_feature = np.zeros(self._node_feature_dim) #node representation
        node_feature[1] = 1 #Interpolation
        node_feature[3] = 1 #is exit
        normalized_block = (block+1) / 5 #normalize to [0,1]
        assert normalized_block >=0 and normalized_block <= 1, "normalized block not in [0,1]"
        node_feature[4] = normalized_block
        node_feature[5] = layer #first layer of the exit
        node_feature[14] = float(threshold)  
        interpolation_index = self._interpol_options.index(interpol_size)
        for i in range(15, 18):
            node_feature[i] = 1 if i-15==interpolation_index else 0
        return node_feature

    def __create_convolution_node(self, block, layer, kernel, expansion, is_exit:bool):
        """Assumes block and layers start counting from 0"""
        node_feature = np.zeros(self._node_feature_dim) #node representation
        node_feature[0]=1 #set type as CONV
        if is_exit:
            node_feature[3]=1
        normalized_block = (block+1) / self._num_of_blocks #normalize to [0,1]
        assert normalized_block >=0 and normalized_block <= 1, "normalized block not in [0,1]"
        node_feature[4] = normalized_block
        normalized_depth = (layer+1) / self._max_depth
        assert normalized_depth >=0 and normalized_depth <= 1, "normalized depth not in [0,1]"
        node_feature[5] = normalized_depth
        kernel_index = self._kernel_options.index(kernel)

        #create one hot encodings
        for i in range(6,9):
            node_feature[i] = 1 if i-6 ==kernel_index else 0
        expansion_index = self._expansion_options.index(expansion)
        for i in range(9, 14):
            node_feature[i] = 1 if i-9==expansion_index else 0
        return node_feature

    def __create_pooling_node(self, block, layer, threshold):
        """Assumes block and layers start counting from 0"""
        node_feature = np.zeros(self._node_feature_dim) #node representation
        node_feature[2]=1 #set type as POOL
        node_feature[3]=1 #is exit
        normalized_block = (block+1) / self._num_of_blocks #normalize to [0,1]
        assert normalized_block >=0 and normalized_block <= 1, "normalized block not in [0,1]"
        node_feature[4] = normalized_block
        normalized_depth = (layer+1) / self._max_depth
        assert normalized_depth >=0 and normalized_depth <= 1, "normalized depth not in [0,1]"
        node_feature[5] = normalized_depth
        node_feature[14] = float(threshold) 
        return node_feature