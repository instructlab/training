import torch

def prepare_fa2_from_position_ids(query, key, value, position_ids, query_length):
    query = query.view(-1, query.size(-2), query.size(-1))
    key = key.view(-1, key.size(-2), key.size(-1))
    value = value.view(-1, value.size(-2), value.size(-1))
    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)
    cu_seq_lens = torch.cat((
        indices_q[position_ids==0],
        torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32)
        ))
    max_length = position_ids.max()+1
    return (query, key, value, indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))