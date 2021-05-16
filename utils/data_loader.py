import random
import paddle


def DataLoader(dataset, batch_size, shuffle=True, drop_last=False):
    # get the dict names
    new_data = dict()
    all_keys = dataset[0].keys()
    for key in all_keys:
        new_data[key] = []
    def reader():
        # get the index list for shuffle
        index_list = list(range(len(dataset)))
        if shuffle:
            random.shuffle(index_list)
        for i in index_list:
            for key in all_keys:
                new_data[key].append(dataset[i][key].unsqueeze(0))
            # a batch
            if len(new_data[key]) == batch_size:
                for key in all_keys:
                    new_data[key] = paddle.concat(new_data[key])
                yield new_data
                for key in all_keys:
                    new_data[key] = []
        # a mini batch
        if len(new_data[key]) > 0:
            if drop_last:
                pass
            else:
                for key in all_keys:
                    new_data[key] = paddle.concat(new_data[key])
                yield new_data
    return reader
