from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.padding.state_padding import StatePadding
from utils.beam_search import beam_search


class PADDING(object):
    NAME = 'padding'

    @staticmethod
    def get_costs(dataset, pi):
        # Check that tours are valid, i.e. contain 0 to n -1
        assert (
                torch.arange(pi.size(1), out=pi.data.new()).view(1, -1).expand_as(pi) == 
                pi.data.sort(1)[0]
        ).all(), "Invalid tour"

        n_events = dataset.size(-2)
        # Gather dataset in order of tour
        sorted_data = dataset.gather(1, pi.unsqueeze(-1).expand_as(dataset))  # reorder row according to pi

        # Get the original features
        t = sorted_data[:, :, [0]]
        is_arrival = sorted_data[:, :, [1]]
        is_departure = sorted_data[:, :, [2]]
        one_hot_cat = sorted_data[:, :, 3:6]
        turn_time_by_cat = sorted_data[:, :, 6:9]

        # Get the new features (feature_1 and feature_2)
        feature_1 = sorted_data[:, :, 9:10]  # 新特征 1
        feature_2 = sorted_data[:, :, 10:11]  # 新特征 2

        is_arrival_departure = is_arrival.matmul(is_departure.transpose(1, 2))
        turn_time_by_connection = one_hot_cat.matmul(turn_time_by_cat.transpose(1, 2))
        dt = t.transpose(1, 2) - t
        all_connection_legality = (dt >= turn_time_by_connection).int()

        lost_connection = all_connection_legality.tril(-1) * is_arrival_departure
        illegal_connection = (1 - all_connection_legality).triu(1) * is_arrival_departure
        if illegal_connection.sum() > 0:
            print("NO!!!!!!! Illegal connection!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return lost_connection.sum(dim=(1, 2)) + 1000.0 * illegal_connection.sum(dim=(1, 2)), None

    @staticmethod
    def make_dataset(*args, **kwargs):
        return PADDINGDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StatePadding.initialize(*args, **kwargs)

    @staticmethod
    def beam_search(input, beam_size, expand_size=None,
                    compress_mask=False, model=None, max_calc_batch_size=4096):
        assert model is not None, "Provide model"

        fixed = model.precompute_fixed(input)

        def propose_expansions(beam):
            return model.propose_expansions(
                beam, fixed, expand_size, normalize=True, max_calc_batch_size=max_calc_batch_size
            )

        state = PADDING.make_state(
            input, visited_dtype=torch.int64 if compress_mask else torch.uint8
        )

        return beam_search(state, beam_size, propose_expansions)


class PADDINGDataset(Dataset):
    def __init__(self, filename=None, num_events=20, num_samples=100000, num_category=3, time_window_length=100,
                 max_turn_time=10, offset=0):
        super(PADDINGDataset, self).__init__()

        self.data_set = []
        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]
        else:
            # adjacent matrix+[is arrival,is departure]
            cat_label = torch.randint(num_category, (num_samples, num_events, 1))
            one_hot_cat = torch.zeros(num_samples, num_events, num_category).scatter_(2, cat_label, 1)
            event_type = torch.randint(2, (num_samples, num_events, 1))
            one_hot_event_type = torch.zeros(num_samples, num_events, 2).scatter_(2, event_type, 1)

            t = torch.rand(num_samples, num_events, 1)
            turn_time = (max_turn_time / time_window_length) * torch.rand(num_samples, num_events, 3)

            # 新增两个特征
            feature_1 = torch.rand(num_samples, num_events, 1)  # 新特征 1
            feature_2 = torch.rand(num_samples, num_events, 1)  # 新特征 2

            # 构建数据：时间 + 到达/离开 + 类别 + 转弯时间 + 新特征
            data = torch.cat([t, one_hot_event_type, one_hot_cat, turn_time, feature_1, feature_2], dim=2)
            self.data = [data[i] for i in range(num_samples)]

        self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
