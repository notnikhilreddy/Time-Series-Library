from data_provider_reservoir.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom
from data_provider_reservoir.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'electricity': Dataset_Custom,
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'traffic': Dataset_Custom,
    'weather': Dataset_Custom,
    'illness': Dataset_Custom,
}

def data_provider(args, flag):
    Data = data_dict[args.data]

    root_path=''
    features='M'
    target='OT'
    scale=True
    timeenc=0
    freq='t'
    num_workers=1

    shuffle_flag = False if flag == 'test' else True
    drop_last = False
    batch_size = args.batch_size

    data_set = Data(
        root_path=root_path,
        flag=flag,
        size=args.size,
        features=features,
        data_path=args.data_path,
        target=target,
        scale=scale,
        timeenc=timeenc,
        freq=freq,
    )
    print(data_set)
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=num_workers,
        drop_last=drop_last)
    return data_set

