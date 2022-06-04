import pickle

char_dir = '/home/server15/sohee_workspace/font_generation/DG-Font/korean_frequent.txt'
file_object = open(char_dir,encoding='utf-8')
characters = file_object.read().split(' ')

dict = {v: k for v, k in enumerate(characters)}

with open('./font_freq_train/char_idx.pkl', 'wb') as handle:
    pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


# with open('/home/server15/sohee_workspace/font_generation/DG-Font/font_freq_train/font_idx.pkl', 'rb') as handle:
#     font_idx = pickle.load(handle)
# for f in font_idx.keys():
#     if f == '나눔손글씨 다채사랑.ttf':
#         print(font_idx[f])

