import pickle
import pandas

# MSVD数据集text质量很差，每个视频text数在18~81间
# 有‘na’、‘disable by request’等无效数据，且绝大部分重复
# 每个视频只取第一个句子作为其描述
info = pickle.load(  # 字典: 1970 {video_name: [[单词], ]}
    open('D:/Code/Python/Graduation_Project/res/msvd_data/raw-captions.pkl', 'rb'))
video_names = list()  # 视频文件名
sentences = list()  # 视频的句子
for (video_name, sentence) in info.items():
    video_names.append(video_name)
    sentences.append(' '.join(sentence[0]).strip().lower())

video_names_train = open('D:/Code/Python/Graduation_Project/res/msvd_data/train_list.txt', 'r', encoding='utf-8'
                         ).read().split('\n') \
                    + open('D:/Code/Python/Graduation_Project/res/msvd_data/val_list.txt', 'r', encoding='utf-8'
                           ).read().split('\n')
video_names_test = open('D:/Code/Python/Graduation_Project/res/msvd_data/test_list.txt', 'r', encoding='utf-8'
                        ).read().split('\n')

df_train_data, df_test_data = list(), list()
idx = 0
for video_name in video_names_train:
    df_train_data.append((idx + 10000, video_name, sentences[video_names.index(video_name)]))
    idx += 1

for video_name in video_names_test:
    df_test_data.append((idx + 10000, video_name, sentences[video_names.index(video_name)]))
    idx += 1

pandas.DataFrame(columns=['video_id', 'video_name', 'sentence'], data=df_train_data
                 ).to_csv('D:/Code/Python/Graduation_Project/res/msvd_data/MSVD_train.csv', index=False)
pandas.DataFrame(columns=['video_id', 'video_name', 'sentence'], data=df_test_data
                 ).to_csv('D:/Code/Python/Graduation_Project/res/msvd_data/MSVD_test.csv', index=False)
