#!/usr/bin/env python
# coding: utf-8

import argparse
import sys
import platform
import os
import pathlib
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import timeit
from PIL import Image
import json
import time
import pickle
from tensorflow.keras.models import load_model
import psutil
import inspect
import datetime
import random
#print('os : '+str(platform.platform()))
#print('python version : ' +str(platform.python_version()))
#print('coeurs cpu : '+str(os.cpu_count()))
path1 = './test-model/models'
model_name = os.listdir(path1)
#print(model_name)
list_image_folder = os.listdir('./test-model/images')
#print(list_image_folder)
weight = []
for i in range (0, len(model_name)):
    weight.append(os.path.getsize(path1+'/'+model_name[i]))  
#print(weight)

parser = argparse.ArgumentParser(description='Choose your model, test, number of test and if you want to save the results.')

parser.add_argument(action = 'store', dest='model_id', nargs =1, type=str,
                    help='Choose the model to evaluate : '+str(model_name))
                    
parser.add_argument(action = 'store', dest='type_test_id', nargs =1, type=str,
                    help='Choose the type of test : '+str(list_image_folder))

parser.add_argument(action = 'store', dest='nb_test_id', nargs =1, type=int,
                    help='Choose the number of test ')

parser.add_argument(action = 'store', dest='status_save', nargs =1, type=str,
                    help='Choose if the test must be saved or not ( s to save, t to not save )')


args = parser.parse_args()
nb_model = args.model_id[0]
test_type = args.type_test_id[0]
nb_test = args.nb_test_id[0]
status_save = args.status_save[0]

model_count = -1
for w in range (0, len(model_name)):
    if model_name[w] == nb_model:
        model_count = w

nb_model = model_count

if nb_model == -1:
    sys.exit('wrong model value')

type_test_count = -1
for z in range (0, len(list_image_folder)):
    if list_image_folder[z] == test_type:
        type_test_count = z

test_type = type_test_count

if test_type == -1:
    sys.exit('wrong type of test value')
    
if nb_test < 1 :
    sys.exit('wrong number of test value')
    
if status_save != 's' and status_save != 't' :
    sys.exit('wrong save status')


def loadModel(name):
    model = load_model(path1+"/"+str(name), compile = False)
    return model

def generate_test(test_type):
    image= None
    path = './test-model/images/'+list_image_folder[test_type]+'/'
    list_image = os.listdir(path)
    path2 = list_image[random.randint(0, len(list_image)-1)]
    image = Image.open(path+path2)
    print(path+path2)  
    return image


def predictModel(model, test_type):
    pred = None
    image = generate_test(test_type)
    #pred2 = []
    image = np.asarray(image)
    image = image.astype('float32')
    image /= 255.0
    image = np.expand_dims(image, 0)
    pred = model.predict(image)
     #for i in range (0,len(pred)):
        #pred[i]=np.argmax(pred[i])
        #pred2.append(int(pred[i][0]))
    #pred = np.asarray(pred2)
    return pred

def get_memory():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss

def get_cpu_percent():
    value = psutil.cpu_percent()
    return value    

def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

# In[3]:

predmodel = inspect.getsource(predictModel)
loadm = inspect.getsource(loadModel)
gen_test = inspect.getsource(generate_test)
letterb_img = inspect.getsource(letterbox_image)

start_time = time.time()

to_save = None
save_data =[]

memory = []
cpu = []

to_save = datetime.datetime.now()
save_data.append(to_save)
print('heure du test : '+str(to_save))

file = open('./test-model/resultats/os-name.txt','r')
to_save = file.readline().strip()
file.close()
save_data.append(to_save)
print('nom de l\'os : '+str(to_save))

to_save = model_name[nb_model]
save_data.append(to_save)
print('nom du model : '+str(to_save))
to_save = weight[nb_model]
save_data.append(to_save)
print('poids du model : '+str(to_save)+' octets')

to_save = list_image_folder[test_type]
print('type de test : '+str(to_save ))
save_data.append(to_save)

to_save = nb_test
save_data.append(to_save)
print('nombre de test : '+str(to_save))

print('')


loadedModel = loadModel(model_name[nb_model])
cpumen_start_time = time.time()

memory.append(get_memory())
cpu.append(get_cpu_percent())

for i in range(0,nb_test):

    #print(loadedModel)
    result = predictModel(loadedModel, test_type)

    memory.append(get_memory())
    cpu.append(get_cpu_percent())
    
#print(result)
#print(memory)
#print(cpu)
#plt.plot(cpu)
#plt.show()

cpu = np.asarray(cpu)
to_save = cpu[cpu.argmin()]
save_data.append(to_save)
print('cpu min : '+str(to_save)+' %')

to_save = cpu[cpu.argmax()]
save_data.append(to_save)
print('cpu max : '+str(to_save)+' %')

somme = np.sum(cpu ,dtype=np.int64)
#print(somme)
to_save = somme / (nb_test+1)
save_data.append(to_save)
print('cpu moyen : '+str(to_save)+' %')

cpu = np.sort(cpu, axis=None)
#print(resultats)
cpu_premier_quartile = np.quantile(cpu, 0.25)
save_data.append(cpu_premier_quartile)
print('cpu 1er quartile : '+str(cpu_premier_quartile)+' %')

cpu_mediane = np.quantile(cpu, 0.50)
save_data.append(cpu_mediane)
print('cpu median : '+str(cpu_mediane)+' %')

cpu_troisieme_quartile = np.quantile(cpu, 0.75)
save_data.append(cpu_troisieme_quartile)
print('cpu 3eme quartile : '+str(cpu_troisieme_quartile)+' %')

to_save = cpu_troisieme_quartile - cpu_premier_quartile
save_data.append(to_save)
print('cpu ecart interquartile : '+str(to_save)+' %')

to_save = np.var(cpu)
save_data.append(to_save)
print('cpu variance : '+str(to_save)+' %')

to_save = np.std(cpu)
save_data.append(to_save)
print('cpu ecart-type : '+str(to_save)+' %')
print('')

#plt.plot(memory)
#plt.show()

memory = np.asarray(memory)
to_save = memory[memory.argmin()]
save_data.append(to_save)
print('memoire min : '+str(to_save)+' octets')

to_save = memory[memory.argmax()]
save_data.append(to_save)
print('memoire max : '+str(to_save)+' octets')

somme = np.sum(memory ,dtype=np.int64)
#print(somme)
to_save = somme / (nb_test+1)
save_data.append(to_save)
print('memoire moyenne : '+str(to_save)+' octets')

memory = np.sort(memory, axis=None)
#print(resultats)
memory_premier_quartile = np.quantile(memory, 0.25)
save_data.append(memory_premier_quartile)
print('memoire 1er quartile : '+str(memory_premier_quartile)+' octets')

memory_mediane = np.quantile(memory, 0.50)
save_data.append(memory_mediane)
print('memoire mediane : '+str(memory_mediane)+' octets')

memory_troisieme_quartile = np.quantile(memory, 0.75)
save_data.append(memory_troisieme_quartile)
print('memoire 3eme quartile : '+str(memory_troisieme_quartile)+' octets')

to_save = memory_troisieme_quartile - memory_premier_quartile
save_data.append(to_save)
print('memoire ecart interquartile : '+str(to_save)+' octets')

to_save = np.var(memory)
save_data.append(to_save)
print('memoire variance : '+str(to_save)+' octets')

to_save = np.std(memory)
save_data.append(to_save)
print('memoire ecart-type : '+str(to_save)+' octets')
print('')

to_save = time.time()-cpumen_start_time
save_data.append(to_save)
print('durée de l\'évaluation mémoire/cpu : '+str(to_save)+' secondes')
print('')

setup_code = '''
from __main__ import loadedModel
import platform
import os
import pathlib
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import timeit
from PIL import Image
import json
import time
import pickle
from tensorflow.keras.models import load_model
import random
#print('os : '+str(platform.platform()))
#print('python version : ' +str(platform.python_version()))
#print('coeurs cpu : '+str(os.cpu_count()))
path1 = './test-model/models'
#print(folder_content)
model_name = os.listdir(path1)
#print(model_name)
list_image_folder = os.listdir('./test-model/images')
#print(list_image_folder)
weight = []
for i in range (0, len(model_name)):
    weight.append(os.path.getsize(path1+'/'+model_name[i]))  
#print(weight)

'''+loadm+'''
'''+gen_test+'''
'''+predmodel+'''
'''+letterb_img+'''
'''

main_block = '''
result = predictModel(loadedModel, '''+str(test_type)+''')
'''

# Main Block
resultats= []
for i in range (0,nb_test):
    resultats.append(timeit.timeit(setup=setup_code, stmt=main_block, number=1))
resultats = np.asarray(resultats)
#print(resultats)
#plt.plot(resultats)
#plt.show()

to_save = resultats[resultats.argmin()]
save_data.append(to_save)
print('temps min : '+str(to_save)+' secondes')

to_save = resultats[resultats.argmax()]
save_data.append(to_save)
print('temps max : '+str(to_save)+' secondes')

somme = sum(resultats)
save_data.append(somme)
print('temps total : '+str(somme)+' secondes')

to_save = somme / nb_test
save_data.append(to_save)
print('temps moyen : '+str(to_save)+' secondes')

resultats = np.sort(resultats, axis=None)
#print(resultats)
resultats_premier_quartile = np.quantile(resultats, 0.25)
save_data.append(resultats_premier_quartile)
print('temps 1er quartile : '+str(resultats_premier_quartile)+' secondes')

resultats_mediane = np.quantile(resultats, 0.50)
save_data.append(resultats_mediane)
print('temps median : '+str(resultats_mediane)+' secondes')

resultats_troisieme_quartile = np.quantile(resultats, 0.75)
save_data.append(resultats_troisieme_quartile)
print('temps 3eme quartile : '+str(resultats_troisieme_quartile)+' secondes')

to_save = resultats_troisieme_quartile - resultats_premier_quartile
save_data.append(to_save)
print('temps ecart interquartile : '+str(to_save)+' secondes')

to_save = np.var(resultats)
save_data.append(to_save)
print('temps variance : '+str(to_save)+' secondes')        

to_save = np.std(resultats)
save_data.append(to_save)
print('temps ecart-type : '+str(to_save)+' secondes')
print('')

to_save = time.time()-start_time
save_data.append(to_save)
print('durée total du test : '+str(to_save)+' secondes')
print('')
#print(save_data)

if status_save == 's':

    df = pd.read_csv('./test-model/resultats/resultats.csv', sep=',')
    number_of_rows = len(df.index)
    df.loc[number_of_rows] = save_data
    df.to_csv("./test-model/resultats/resultats.csv", sep=',', index=False)

    print('Données sauvegardées')
else:
    print('Données non sauvegardées')