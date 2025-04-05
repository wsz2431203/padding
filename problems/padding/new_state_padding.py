import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter

class StatePadding(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    feature_1: torch.Tensor  # 新特征 1
    feature_2: torch.Tensor  # 新特征 2

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
            lengths=self.lengths[key],
            feature_1=self.feature_1[key],  # 修改
            feature_2=self.feature_2[key]   # 修改
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

        # 新特征的初始化
        feature_1 = torch.rand(batch_size, n_loc, 1, device=loc.device)  # 随机生成特征 1
        feature_2 = torch.rand(batch_size, n_loc, 1, device=loc.device)  # 随机生成特征 2

        return StatePadding(
            loc=loc,
            feature_1=feature_1,  # 添加新特征 1
            feature_2=feature_2,  # 添加新特征 2
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

    def all_finished(self):
        return self.i.item() >= self.loc.size(-2)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited + self.illegal_ > 0

    def get_nn(self, k=None):
        if k is None:
            k = self.loc.size(-2) - self.i.item()
        return (self.dist[self.ids, :, :] + self.visited.float()[:, :, None, :] * 1e6).topk(k, dim=-1, largest=False)[1]

    def get_nn_current(self, k=None):
        assert False, "Currently not implemented, look into which neighbours to use in step 0?"
        if k is None:
            k = self.loc.size(-2)
        return (
                self.dist[self.ids, self.prev_a]
                + self.visited.float() * 1e6
        ).topk(k, dim=-1, largest=False)[1]

    def construct_solutions(self, actions):
        return actions
