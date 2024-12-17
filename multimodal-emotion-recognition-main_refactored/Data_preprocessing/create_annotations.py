# -*- coding: utf-8 -*-

import os
import sys
import random

# Ottieni il percorso della directory superiore
cartella_superiore = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(cartella_superiore)

import opts
opt = opts.parse_opts()

root = "/home/v.mele/cognitive_robotics/datasets/RAVDESS"


n_folds=1
folds = [[[0,1,2,3],[4,5,6,7],[8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]]]
for fold in range(n_folds):
        fold_ids = folds[fold]
        test_ids, val_ids, train_ids = fold_ids
        training_set = []
        validation_set = []
        test_set = []
	
        annotation_file = 'annotations.txt'
	
        for i,actor in enumerate(os.listdir(root)):
            for video in os.listdir(os.path.join(root, actor)):
                if not video.endswith('.npy') or 'croppad' not in video:
                    continue
                label = str(int(video.split('-')[2]))
                target=""
                if label=='1' or label=='2':
                    target=str(1) #Neutral
                elif label=='3' or label=='8':
                    target=str(2) #Happy
                elif label=='5' or label=='6' or label=='7':
                    target=str(3) #Angry
                elif label=='4':
                    target=str(4) #Sad
                audio = '03' + video.split('_face')[0][2:] + '_croppad.wav' 
                if i in train_ids:
                   with open(annotation_file, 'a') as f:
                       f.write(os.path.join(root,actor, video) + ';' + os.path.join(root,actor, audio) + ';' + target + ';training' + '\n')
		

                elif i in val_ids:
                    with open(annotation_file, 'a') as f:
                        f.write(os.path.join(root, actor, video) + ';' + os.path.join(root,actor, audio) + ';'+ target + ';validation' + '\n')
		
                else:
                    with open(annotation_file, 'a') as f:
                        f.write(os.path.join(root, actor, video) + ';' + os.path.join(root,actor, audio) + ';'+ target + ';testing' + '\n')
                


