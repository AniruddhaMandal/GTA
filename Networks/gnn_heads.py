import torch

class InductiveEdge(torch.nn.Module):
    def __init__(self, in_dim, out_dim, edge_decoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.edge_decoder = edge_decoder
        if edge_decoder == "concat":
            self.layers = torch.nn.Sequential(torch.nn.Linear(in_dim*2,in_dim),
                                              torch.nn.Linear(in_dim,out_dim))
        else:
            if out_dim > 1:
                raise ValueError(f"Binary edge decoding ({edge_decoder}) is used for multi-class edge/link prediction.")
            self.layers = torch.nn.Sequential(torch.nn.Linear(in_dim,in_dim),
                                              torch.nn.Linear(in_dim, in_dim))

    def decode_concat(self, v1, v2):
        return self.layers(torch.cat((v1,v2), dim=-1))
    
    def decode_dot(self, v1, v2):
        return torch.sum(v1*v2, dim=-1) 

    def decode_cosine(self, v1, v2):
        return torch.nn.functional.cosine_similarity(v1,v2,dim=-1)

    def forward(self, batch):
        if self.edge_decoder != "concat":
            batch.x = self.layers(batch.x)
        pred = batch.x[batch.edge_label_index]
        nodes_first = pred[0]
        nodes_second = pred[1]
        if self.edge_decoder == "concat":
            score = self.decode_concat(nodes_first, nodes_second)
        if self.edge_decoder == "dot":
            score = self.decode_dot(nodes_first, nodes_second)
        elif self.edge_decoder == "cosine_similarity":
            score = self.decode_cosine(nodes_first, nodes_second)
        else: 
            raise ValueError(f"Unknown edge decoder: {self.edge_decoder}")
        return score

