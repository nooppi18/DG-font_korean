#from msilib import text
from PIL import Image,ImageDraw,ImageFont
import matplotlib.pyplot as plt
import os
import numpy as np
import pathlib
import argparse
import random
import pickle


parser = argparse.ArgumentParser(description='Obtaining characters from .ttf')
parser.add_argument('--ttf_path', type=str, default='../font',help='ttf directory')
parser.add_argument('--chara', type=str, default='../chara.txt',help='characters')
parser.add_argument('--save_path', type=str, default='../src_ttf',help='images directory')
parser.add_argument('--img_size', type=int, help='The size of generated images')
parser.add_argument('--chara_size', type=int, help='The size of generated characters')
args = parser.parse_args()

file_object = open(args.chara,encoding='utf-8')
try:
	characters = file_object.read().split(' ') # split 추가 -> enter 친 txt 파일에서 글자만 생성 가능
finally:
    file_object.close()


def draw_single_char(ch, font, canvas_size, x_offset, y_offset):
    img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    textsize_w = draw.textsize(ch, font=font)[0]
    textsize_h = draw.textsize(ch, font=font)[1]
    x_offset = (canvas_size-textsize_w) / 2 
    y_offset =canvas_size / 2 - textsize_h * 5/8#(canvas_size-textsize_h) / 2

    draw.text((x_offset, y_offset), ch, (0, 0, 0), font=font, align='center')
    return img

def draw_example(ch, src_font, canvas_size, x_offset, y_offset):
    src_img = draw_single_char(ch, src_font, canvas_size, x_offset, y_offset)
    example_img = Image.new("RGB", (canvas_size, canvas_size), (255, 255, 255))
    example_img.paste(src_img, (0, 0))
    return example_img

data_dir = args.ttf_path
data_root = pathlib.Path(data_dir)
print(data_root)

all_image_paths = list(data_root.glob('*.ttf*'))
all_image_paths = [str(path) for path in all_image_paths]
print(len(all_image_paths))
for i in range (len(all_image_paths)):
    print(all_image_paths[i])

print(random.sample(all_image_paths, 2)) #6


seq = list()

print('start making font_images')
font_idx = {}
for (label,item) in zip(range(len(all_image_paths)),all_image_paths):
    print(item, label)
    font_idx[item.split('/')[-1]] = label
    src_font = ImageFont.truetype(item, size = args.chara_size)
    for (chara,cnt) in zip(characters, range(len(characters))):
       

        img = draw_example(chara, src_font, args.img_size, args.img_size/2, (args.img_size-args.chara_size)/2)#(args.img_size-args.chara_size)/2, (args.img_size-args.chara_size)/2)
        path_full = os.path.join(args.save_path, 'id_%d'%label)
        if not os.path.exists(path_full):
            os.mkdir(path_full)
        img.save(os.path.join(path_full, "%04d.png" % (cnt)))
        
with open(f'{args.save_path}/font_idx.pkl', 'wb') as handle:
    pickle.dump(font_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)