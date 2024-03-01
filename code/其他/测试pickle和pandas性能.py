import numpy
import torch
import pickle
import pandas
import time

index = list(range(10000))
data1 = torch.randn(10000, 1, 512, dtype=torch.float)
data2 = torch.randn(10000, 12, 512, dtype=torch.float)


def store():
    # 直接保存torch.tensor：1.906s 253MB
    # pickle.dump((numpy.array(index), data1, data2), open('pickle.pkl', 'wb'))
    # a, b, c = pickle.load(open('pickle.pkl', 'rb'))
    # return a, b, c

    # 直接保存numpy.array：0.828s 253MB
    # pickle.dump((numpy.array(index), data1.numpy(), data2.numpy()), open('pickle.pkl', 'wb'))
    # a, b, c = pickle.load(open('pickle.pkl', 'rb'))
    # return a, torch.tensor(b), torch.tensor(c)

    # Pandas DataFrame保存torch.tensor为pickle：∞ ≥25.1GB 运行，MemoryError终止
    # df = pandas.DataFrame(columns=['video_id', 'text_features', 'video_features'],
    #                       data=zip([i for i in numpy.array(index)], [i for i in data1], [i for i in data2])
    #                       ).set_index('video_id')
    # df.to_pickle('pandas.pkl')
    # df = pandas.read_pickle('pandas.pkl').reset_index(drop=False)
    # a = df['video_id'].values
    # b = torch.vstack(df['text_features'])
    # c = torch.vstack(df['video_features'])
    # return a, b, c

    # Pandas DataFrame保存numpy.array为pickle：0.718s 254MB
    df = pandas.DataFrame(columns=['video_id', 'text_features', 'video_features'],
                          data=zip([i for i in numpy.array(index)], [i for i in data1.numpy()], [i for i in data2.numpy()])
                          ).set_index('video_id')
    df.to_pickle('pandas.pkl')
    df = pandas.read_pickle('pandas.pkl').reset_index(drop=False)
    a = df['video_id'].values
    b = torch.tensor(numpy.stack(df['text_features'], axis=0))
    c = torch.tensor(numpy.stack(df['video_features'], axis=0))
    return a, b, c


time1 = time.time()
_index, _data1, _data2 = store()
print(type(_index), type(_data1), type(_data2))
print(len(_index), _index[:3])
print(_data1.shape, _data2.shape)
print(_data1[1].squeeze() @ _data2[1, 1])
print(f'共用时{time.time() - time1}s')
