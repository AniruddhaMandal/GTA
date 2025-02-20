from Networks.message_passing_lstm import MessagePassingLSTM
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

    if(cfg.Model.framework == "message-passing-lstm"):
        return MessagePassingLSTM(in_dim=in_dim, 
                                  out_dim=out_dim, 
                                  hidden_dim=hidden_dim,
                                  hops=hops,
                                  dropout=dropout_frac,
                                  type=type,
                                  encoder=encoder)
    
    if(cfg.Model.framework == "vanilla"):
        return Vanilla(in_dim=in_dim,
                       out_dim=out_dim,
                       hidden_dim=hidden_dim,
                       hops=hops,
                       dropout=dropout_frac,
                       type=type,
                       encoder=encoder)