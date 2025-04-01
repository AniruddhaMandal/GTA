from Networks.vanilla import Vanilla

def build_model(cfg):
    """ Handels model type. Sets up the model 
        configuration. Returns configured model."""
    in_dim = cfg.Data.input_dim
    out_dim = cfg.Data.output_dim
    hidden_dim = cfg.Model.hidden_dim
    hops = cfg.Model.hops
    dropout_frac = cfg.Model.dropout_frac
    type = cfg.Model.type
    encoder = cfg.Model.encoder
    graph_pooling = cfg.Model.graph_pooling 
    edge_encoder = cfg.Model.edge_encoder

    # Add Models Here
    if(cfg.Model.framework == "vanilla"):
        return Vanilla(in_dim=in_dim,
                       out_dim=out_dim,
                       hidden_dim=hidden_dim,
                       hops=hops,
                       dropout=dropout_frac,
                       type=type,
                       encoder=encoder,
                       graph_pooling=graph_pooling,
                       edge_encoder=edge_encoder)