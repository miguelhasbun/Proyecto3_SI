#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 07:16:04 2018

@author: raghav prabhu
Re-modified TensorFlow classification file according to our need.
"""
import tensorflow as tf
import sys
import os
import csv
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


# Disable tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

'''
Classify images from test folder and predict dog breeds along with score.
'''


def classify_image(image_path, headers, file_name, img):
    f = open('submit.csv','w')
    writer = csv.DictWriter(f, fieldnames = headers)
    writer.writeheader()
    
    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                   in tf.gfile.GFile("trained_model/retrained_labels.txt")]
   
    # Unpersists graph from file
    with tf.gfile.FastGFile("trained_model/retrained_graph.pb", 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    files = os.listdir(image_path)
    with tf.Session() as sess:
         for file in files:
             # Read the image_data
                image_data = tf.gfile.FastGFile(image_path+'/'+file, 'rb').read()
                # Feed the image_data as input to the graph and get first prediction
                softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

                predictions = sess.run(softmax_tensor, \
                                       {'DecodeJpeg/contents:0': image_data})

                # Sort to show labels of first prediction in order of confidence
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                records = []
                row_dict = {}
                head, tail = os.path.split(file)
                row_dict['id'] = tail.split('.')[0]

                for node_id in top_k:
                    human_string = label_lines[node_id]

                    # Some breed names are mismatching with breed name in csv header names.
                    human_string = human_string.replace(" ","_")
                    if(human_string == 'german_short_haired_pointer'):
                        human_string = 'german_short-haired_pointer'
                    if(human_string == 'shih_tzu'):
                        human_string = 'shih-tzu'
                    if(human_string == 'wire_haired_fox_terrier'):
                        human_string = 'wire-haired_fox_terrier'
                    if(human_string == 'curly_coated_retriever'):
                        human_string = 'curly-coated_retriever'
                    if(human_string == 'black_and_tan_coonhound'):
                        human_string = 'black-and-tan_coonhound'
                    if(human_string == 'soft_coated_wheaten_terrier'):
                        human_string = 'soft-coated_wheaten_terrier'  
                    if(human_string == 'flat_coated_retriever'):
                        human_string = 'flat-coated_retriever'    
                    score = predictions[0][node_id]
                    print('%s (score = %.5f)' % (human_string, score))
                    row_dict[human_string] = score

                human_string = label_lines[top_k[0]]
                score2 = predictions[0][top_k[0]]
                image_text = human_string + "\nscore = " + str(score2)
                imagen = Image.open(img)
                draw = ImageDraw.Draw(imagen)
                font = ImageFont.truetype("arial.ttf", 25)
                draw.text((0, 0), image_text, (255, 255, 255), font=font)
                imagen.save(file_name)
                imagen.show()

                records.append(row_dict.copy())
                writer.writerows(records)
    f.close()    

def main():
    full_path = ''
    file_name = ''
    command = ''

    while command!= 'exit':
        command = input('console>>')
        if command == 'exit':
            continue
        elif command == 'image':
            full_path = input('Ingrese el full path a la imagen: ')
            file_name = input('Nombre de archivo con su extension: ')
        else:
            print('Unknown command. Please write '"image"' or '"exit"'')
            continue

        img = full_path + '/'+file_name
        test_data_folder = full_path +"/"
        template_file = open('sample_submission.csv','r')
        d_reader = csv.DictReader(template_file)

        #get fieldnames from DictReader object and store in list
        headers = d_reader.fieldnames
        template_file.close()
        classify_image(test_data_folder, headers, file_name, img)
    

if __name__ == '__main__':
    main()
