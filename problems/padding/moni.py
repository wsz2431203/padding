import torch
from typing import NamedTuple

import numpy as np
import torch.nn.functional as F

def _pad_mask(mask):
    # By taking -size % 8, we get 0 if exactly divisible by 8
    # and required padding otherwise (i.e. -1 % 8 = 7 pad)
    pad = -mask.size(-1) % 8
    if pad != 0:
        mask = F.pad(mask, [0, pad])
    return mask, mask.size(-1) // 8


def _mask_bool2byte(mask):
    assert mask.dtype == torch.uint8
    # assert (mask <= 1).all()  # Precondition, disabled for efficiency
    mask, d = _pad_mask(mask)
    return (mask.view(*mask.size()[:-1], d, 8) << torch.arange(8, out=mask.new())).sum(-1, dtype=torch.uint8)


def _mask_byte2long(mask):
    assert mask.dtype == torch.uint8
    mask, d = _pad_mask(mask)
    # Note this corresponds to a temporary factor 8
    # memory overhead by converting to long before summing
    # Alternatively, aggregate using for loop
    return (mask.view(*mask.size()[:-1], d, 8).long() << (torch.arange(8, dtype=torch.int64, device=mask.device) * 8)).sum(-1)


def mask_bool2long(mask):
    assert mask.dtype == torch.uint8
    return _mask_byte2long(_mask_bool2byte(mask))


def _mask_long2byte(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] >> (torch.arange(8, out=mask.new()) * 8))[..., :n].to(torch.uint8).view(*mask.size()[:-1], -1)[..., :n]


def _mask_byte2bool(mask, n=None):
    if n is None:
        n = 8 * mask.size(-1)
    return (mask[..., None] & (mask.new_ones(8) << torch.arange(8, out=mask.new()) * 1)).view(*mask.size()[:-1], -1)[..., :n] > 0


def mask_long2bool(mask, n=None):
    assert mask.dtype == torch.int64
    return _mask_byte2bool(_mask_long2byte(mask), n=n)


def mask_long_scatter(mask, values, check_unset=True):
    """
    Sets values in mask in dimension -1 with arbitrary batch dimensions
    If values contains -1, nothing is set
    Note: does not work for setting multiple values at once (like normal scatter)
    """
    assert mask.size()[:-1] == values.size()
    rng = torch.arange(mask.size(-1), out=mask.new())
    values_ = values[..., None]  # Need to broadcast up do mask dim
    # This indicates in which value of the mask a bit should be set
    where = (values_ >= (rng * 64)) & (values_ < ((rng + 1) * 64))
    # Optional: check that bit is not already set
    assert not (check_unset and ((mask & (where.long() << (values_ % 64))) > 0).any())
    # Set bit by shifting a 1 to the correct position
    # (% not strictly necessary as bitshift is cyclic)
    # since where is 0 if no value needs to be set, the bitshift has no effect
    return mask | (where.long() << (values_ % 64))



class StatePadding(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    # dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    illegal_: torch.Tensor  # Keeps track of nodes that definitely cause illegal connection if visit next
    curr_sol: torch.Tensor
    curr_pos: torch.Tensor  # the position of each event in current solution
    all_connection_legality: torch.Tensor
    is_arrival_departure: torch.Tensor
    lengths: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.uint8:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    def __getitem__(self, key):
        assert torch.is_tensor(key) or isinstance(key, slice)  # If tensor, idx all tensors by this tensor:
        return self._replace(
            ids=self.ids[key],
            first_a=self.first_a[key],
            prev_a=self.prev_a[key],
            visited_=self.visited_[key],
            lengths=self.lengths[key]
        )

    @staticmethod
    def initialize(loc, visited_dtype=torch.uint8):

        batch_size, n_loc, _ = loc.size()
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        t = loc[:, :, [0]]
        is_arrival = loc[:, :, [1]]
        is_departure = loc[:, :, [2]]
        one_hot_cat = loc[:, :, 3:6]
        turn_time_by_cat = loc[:, :, 6:9]
        is_arrival_departure = is_arrival.matmul(is_departure.transpose(1, 2))
        turn_time_by_connection = one_hot_cat.matmul(turn_time_by_cat.transpose(1, 2))
        dt = t.transpose(1, 2) - t
        all_connection_legality = (dt >= turn_time_by_connection).int()
        illegal = ((1 - all_connection_legality) * is_arrival_departure).max(dim=-1)[0].unsqueeze(-2)
        return StatePadding(
            loc=loc,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
            first_a=prev_a,
            prev_a=prev_a,
            visited_=torch.zeros(
                batch_size, 1, n_loc,
                dtype=torch.uint8, device=loc.device),
            illegal_=illegal,
            curr_sol=torch.arange(n_loc, dtype=torch.int64, device=loc.device).unsqueeze(0).expand(batch_size, -1),
            curr_pos=torch.arange(n_loc, dtype=torch.int64, device=loc.device).unsqueeze(0).expand(batch_size, -1),
            all_connection_legality=all_connection_legality,
            is_arrival_departure=is_arrival_departure,
            lengths=torch.zeros(batch_size, 1, device=loc.device),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()
        # assert self.visited_.

        return self.lengths

    def update(self, selected):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step
        lengths = self.lengths
        i = self.i

        all_connection_legality = self.all_connection_legality
        is_arrival_departure = self.is_arrival_departure

        curr_sol = self.curr_sol
        curr_pos = self.curr_pos
        # swap two columns
        selected_pos = torch.zeros(selected.size(0), 2, dtype=torch.int64, device=self.loc.device)
        selected_event = torch.zeros(selected.size(0), 2, dtype=torch.int64, device=self.loc.device)
        selected_pos[:, 0] = i
        selected_pos[:, 1] = curr_pos.gather(1, selected.unsqueeze(1)).squeeze(1)
        selected_event[:, 0] = curr_sol[:, i.item()]
        selected_event[:, 1] = selected

        curr_sol = curr_sol.scatter(1, selected_pos, curr_sol.gather(1, selected_pos).roll(1, dims=-1))
        curr_pos = curr_pos.scatter(1, selected_event, curr_pos.gather(1, selected_event).roll(1, dims=-1))

        expanded_swap_ids = selected_pos.unsqueeze(-2).expand(-1, all_connection_legality.size(-1), -1)
        all_connection_legality = all_connection_legality.scatter(2, expanded_swap_ids, all_connection_legality
                                                                  .gather(2, expanded_swap_ids).roll(1, dims=-1))
        is_arrival_departure = is_arrival_departure.scatter(2, expanded_swap_ids, is_arrival_departure
                                                            .gather(2, expanded_swap_ids).roll(1, dims=-1))

        # swap two rows
        expanded_swap_ids = selected_pos.unsqueeze(-1).expand(-1, -1, all_connection_legality.size(-1))
        all_connection_legality = all_connection_legality.scatter(1, expanded_swap_ids, all_connection_legality
                                                                  .gather(1, expanded_swap_ids).roll(1, dims=1))
        is_arrival_departure = is_arrival_departure.scatter(1, expanded_swap_ids, is_arrival_departure
                                                            .gather(1, expanded_swap_ids).roll(1, dims=1))

        illegal = torch.zeros_like(self.visited_, dtype=torch.uint8, device=self.loc.device)
        if i + 1 < self.loc.size(-2):
            illegal = (1 - all_connection_legality[:, :, i + 1:]) * is_arrival_departure[:, :, i + 1:]
            illegal = illegal.max(dim=-1)[0].gather(1, curr_pos).unsqueeze(-2)

        new_lost_connection = all_connection_legality[:, i, :i] * is_arrival_departure[:, i, :i]
        new_illegal_connection = (1 - all_connection_legality[:, :i, i]) \
                                 * is_arrival_departure[:, :i, i]
        if new_illegal_connection.sum() > 0:
            print("=====================================!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        lengths += new_lost_connection.sum(dim=-1) + 1000.0 * new_illegal_connection.sum(dim=-2)

        first_a = prev_a if self.i.item() == 0 else self.first_a

        if self.visited_.dtype == torch.uint8:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited_, illegal_=illegal,
                             curr_sol=curr_sol, curr_pos=curr_pos,
                             lengths=lengths, all_connection_legality=all_connection_legality,
                             is_arrival_departure=is_arrival_departure, i=i + 1)

    def wrong_update(self, selected):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step
        visited = self.visited_

        # is_illegal_arrival_departure = (1 - all_connection_legality) * is_arrival_departure
        # is_legal_arrival_departure = all_connection_legality * is_arrival_departure
        #
        # new_lost_connection = all_connection_legality[:, i, :i] * is_arrival_departure[:, i, :i]
        # new_illegal_connection = self.illegal_.gather(2, prev_a.unsqueeze(-1))
        # lengths += new_lost_connection.sum(dim=-1) + 1000.0 * new_illegal_connection.sum(dim=-2)

        first_a = prev_a if self.i.item() == 0 else self.first_a

        visited = visited.scatter(-1, prev_a[:, :, None], 1)

        # illegal = (1 - visited.int()).matmul(is_illegal_arrival_departure)

        return self._replace(first_a=first_a, prev_a=prev_a, visited_=visited, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i.item() >= self.loc.size(-2)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited + self.illegal_ > 0  # Hacky way to return bool or uint8 depending on pytorch version

    def get_nn(self, k=None):
        # Insert step dimension
        # Nodes already visited get inf so they do not make it
        if k is None:
            k = self.loc.size(-2) - self.i.item()  # Number of remaining
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    def get_nn_current(self, k=None):
        assert False, "Currently not implemented, look into which neighbours to use in step 0?"
        # Note: if this is called in step 0, it will have k nearest neighbours to node 0, which may not be desired
        # so it is probably better to use k = None in the first iteration
        if k is None:
            k = self.loc.size(-2)
        k = min(k, self.loc.size(-2) - self.i.item())  # Number of remaining
        return (
                self.dist[
                    self.ids,
                    self.prev_a
                ] +
                self.visited.float() * 1e6
        ).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions



import torch
import numpy as np

# 生成模拟数据
def generate_event_data(n_events=10):
    # 时间戳（升序排列）
    timestamps = np.sort(np.random.uniform(0, 100, n_events))
    
    # 是否是到达事件(0/1)
    is_arrival = np.random.randint(0, 2, n_events)
    is_departure = 1 - is_arrival
    
    # 类别（3个类别）
    categories = np.random.randint(0, 3, n_events)
    one_hot_cat = np.eye(3)[categories]
    
    # 每个类别的转换时间
    turn_times = np.random.uniform(0, 60, (n_events, 3))
    
    # 组装数据
    loc_data = np.column_stack([
        timestamps,  # 时间戳
        is_arrival,  # 是否到达
        is_departure,  # 是否出发
        one_hot_cat,  # one-hot类别
        turn_times  # 转换时间
    ])
    
    return torch.tensor(loc_data, dtype=torch.float).unsqueeze(0)

def simulate_event_selection(loc):
    # 初始化状态
    state = StatePadding.initialize(loc)
    
    print("初始状态:")
    print(f"当前解决方案: {state.curr_sol}")
    print(f"当前位置: {state.curr_pos}")
    print(f"非法连接: {state.illegal_}")
    
    # 模拟选择过程
    selections = []
    while not state.all_finished():
        # 获取可选择的节点掩码（形状为 [batch_size, 1, n_loc]）
        mask = state.get_mask()
        
        # 找出当前batch的第一个样本的可用节点（假设batch_size=1）
        available_nodes = torch.where(mask[0, 0] == 0)[0]  # 取第一个样本的mask
        
        if len(available_nodes) == 0:
            print("没有可选择的节点!")
            break
        
        # 随机选择一个可用节点（确保是标量张量）
        selected_idx = torch.randint(len(available_nodes), (1,))
        selected = available_nodes[selected_idx]
        
        # 转换为适合update方法的格式：[batch_size] 的LongTensor
        selected = selected.view(1).to(state.loc.device)
        selections.append(selected.item())
        
        print(f"\n准备选择的节点: {selected.item()}")
        print(f"选择前的解决方案: {state.curr_sol}")
        
        # 更新状态
        state = state.update(selected)
        
        print(f"步骤 {state.i.item()}:")
        print(f"选择后的解决方案: {state.curr_sol}")
        print(f"当前位置映射: {state.curr_pos}")
        print(f"累计惩罚: {state.lengths.item()}")
    
    print("\n最终结果:")
    print(f"选择顺序: {selections}")
    print(f"最终解决方案: {state.curr_sol}")
    print(f"总代价: {state.get_final_cost().item()}")
    
    return state, selections

# 运行模拟（添加了更详细的打印信息）
if __name__ == "__main__":
    torch.manual_seed(42)  # 固定随机种子便于调试
    print("生成测试数据...")
    loc = generate_event_data(5)  # 使用较少数目便于观察
    print("生成的数据形状:", loc.shape)
    print("数据内容:\n", loc)
    
    print("\n开始模拟选择过程...")
    final_state, selections = simulate_event_selection(loc)
    
    # 验证最终解的有效性
    print("\n验证结果:")
    solution = final_state.curr_sol[0].tolist()
    print("节点顺序:", solution)
    print("是否所有节点都被访问:", torch.all(final_state.visited[0, 0]))