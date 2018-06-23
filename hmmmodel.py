from __future__ import division
import re
import pandas as pd
import codecs
import os
from os.path import basename
import unicodedata
import numpy as np
import collections
import json
import argparse
from nltk.tag import pos_tag
import regex
from nltk.tree import Tree
from nltk.chunk import ne_chunk
from collections import Counter
import sys
from geotext import GeoText
import string
import xml.etree.ElementTree as ET
import lxml
from lxml import etree
import itertools

#import nltk.data
from nltk import tokenize
import pdb
import glob
import nameparser
import nltk
import markovify
import pathos
import timeit
from nameparser.parser import HumanName

import numpy as np
from sklearn import metrics

import nltk


from nltk import ne_chunk, pos_tag
from nltk.tokenize import word_tokenize
from nltk.tree import Tree

from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
from hmmlearn import hmm


'''
def get_continuous_chunks(text):
    chunked = ne_chunk(pos_tag(word_tokenize(text)))
    prev = None
    continuous_chunk = []
    current_chunk = []
    label_chunk = []

    for i in chunked:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
            label_chunk.append(i.label())

        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return zip(continuous_chunk,label_chunk)

txt = 'The new GOP era in Washington got off to a messy start Tuesday as House Republicans,under pressure from President-elect Donald Trump.'

print (get_continuous_chunks(txt))

'''


# step 1 Data Preparation
#Input Raw text output Annotated text(tagged text)

text = 'While in France, Christine Lagarde discussed short-term stimulus efforts in a recent interview with the Wall Street Journal.'

train_data=["Sameer is a intelligent boy",
            "Deepa lives in Nagpur",
            "Ankit is a football player",
            "Aabhas plays cricket"
            ]


st = StanfordNERTagger('/usr/share/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz',
					   '/usr/share/stanford-ner/stanford-ner.jar',
					   encoding='utf-8')


def namedent(sent):
    tokenized_text = word_tokenize(sent)
    classified_text = st.tag(tokenized_text)
    return classified_text


chunked_text = map(namedent,train_data)
print (chunked_text)
print (train_data)


# step 2 Parameter Estimation (Training)

# Procedure to find the states

l = ([item for sublist in ([[str(tag) for word,tag in i] for i in chunked_text]) for item in sublist])

print (l)


state = []

for m in l:
    if m not in state:
        state.append(m)
print (state)



# Procedure to find the start Probability
wlist = []
start_pblist = []
for w in state:
    # [[str(tag) for word,tag in i][0] for i in chunked_text]
    start_pair =  ([[str(tag) for word,tag in i][0] for i in chunked_text].count(w))/len(train_data)
    print (start_pair)
    wlist.append(w)
    start_pblist.append(start_pair)
start_pb = zip(wlist,start_pblist)
start_pb = dict(start_pb)
print (start_pb)
print type(start_pb)

print (chunked_text)

result1 = {}
for i in chunked_text:
    # pair_count = [i.count(w) for w in i]
    # pair = [w for w in i]
    # emiss_pbnr = dict(zip(pair, pair_count))
    for m in i:
        print m
        emiss_pbnr = m
        if result1.has_key(emiss_pbnr):
            result1[emiss_pbnr]=result1[emiss_pbnr]+1
        else:
            result1[emiss_pbnr]=1
            print emiss_pbnr
    
freq_word = [l.count(i) for i in state]
print (freq_word)
print (state)
emiss_pbdr= dict(zip(state, freq_word))
print (emiss_pbdr)
print (result1)

# Procedure to find the Transition Probability

print (chunked_text)

count = 0
count1 = 0
trans_pb={}

for i in chunked_text:
    n = ([str(tag) for word,tag in [w for w in i]])
    print n
    st = "start"
    en = "end"
    n = [st] + n + [en]
    print n
    for i in range(0, len (n)):
        # print n[i]
        # print n[i+1]
        if n[i] != "end":
            if n[i] != "start":
                if n[i+1] != "end":
                    transpair = (n[i],n[i+1])
                else:
                    transpair = (n[i],n[i])
                if trans_pb.has_key(transpair):
                    trans_pb[transpair]=trans_pb[transpair]+1
                else:
                    trans_pb[transpair]=1
                print trans_pb
newKey1 = []
newValue1 = []
for key,value in trans_pb.iteritems():
    value = round(value /(emiss_pbdr[key[0]]), 6)
    print (key,value)
    newKey1.append(key)
    newValue1.append(value)
    print value    
newKey1,newValue1
trans_pb = dict(zip(newKey1,newValue1))
print (trans_pb)
print (start_pb)



# Procedure to find the Emission Probability
newKey = []
newValue = []
for key,value in result1.iteritems():
    print (key,value)
    print key
    value = round(value / (emiss_pbdr[str(key[-1])]), 6)
    print value
    newKey.append(key)
    newValue.append(value)
newKey = [(str(t[1]), str(t[0])) for t in newKey]
newKey,newValue
emiss_pb = dict(zip(newKey,newValue))
print (emiss_pb)


# print states
# print start_pb
# print trans_pb
# print emiss_pb

# step 3 Test the System

states = []
emissionprob=[]
valuelst= []
observations = []
emissvalue1 = []
for key,value in emiss_pb.iteritems():   
    states.append(key[0])
    # observations.append(str(key[0]))
    pair = key[1],value
    emissionprob.append(pair)
    pair1 = key[1]
    value1 = value
    observations.append(pair1)
    emissvalue1.append(value1)
print (states)
print (emissionprob)
print (observations)
print (emissvalue1)
emiss1 = zip(states,emissionprob)
emiss1 = ([(m,(n[0],n[1])) for m,n in emiss1])
print (emiss1)
print (emiss1)

from itertools import groupby

things = [("animal", "bear"), ("animal", "duck"), ("plant", "cactus"), ("vehicle", "speed boat"), ("vehicle", "school bus")]

for key, group in groupby(things, lambda x: x[0]):
    for thing in group:
        print "A %s is a %s." % (thing[1], key)
    print " "

print (emiss1)

res = {}
for x in emiss1:
    if x[0] in res.keys():
        res[x[0]][x[1][0]] = x[1][1]
    else:
        res[x[0]] = {}
        res[x[0]][x[1][0]] = x[1][1]

emis_pb = res

# print emisspair1

key2list = []
pair2list = []
for key, value in trans_pb.iteritems():
    print key, value
    key2 = key[0]
    pair2 = (key[1], value)
    key2list.append(key2)
    pair2list.append(pair2)
    trans_pblist = zip(key2list,pair2list)
    print (trans_pblist)


print states
print state
print start_pb
print trans_pb
print emis_pb
print (trans_pblist)

res1 = {}

for y in trans_pblist:
    print y
    if y[0] in res1.keys():
        res1[y[0]][y[1][0]] = y[1][1]
    else:
        res1[y[0]] = {}
        res1[y[0]][y[1][0]] = y[1][1]

print res1

tras_pb = res1

print states
print state
print trans_pb

print state
print observations
print (start_pb)
print (emis_pb)
print (tras_pb)



# for key, value in emis_pb.iteritems():
#     print observations
#     print key, value
#     for key3, value3 in value.iteritems():
#         print key3, value3
#         print observations

for key, value in emis_pb.iteritems():
    for observation in observations:
        if value.has_key(observation):
            pass
        else:
            value[observation]=0.0
print emis_pb

for key, value in tras_pb.iteritems():
    for st in state:
        if value.has_key(st):
            pass
        else:
            value[st] = 0.0
print tras_pb

start_valuelist = []
tras_valuelist = []
emis_valuelist = []
emis_valuelist1 = []
tras_valuelist1 = []

for start_key,start_value in start_pb.iteritems():
    start_valuelist.append(start_value)

for tras_key,tras_value in tras_pb.iteritems():
    if tras_valuelist != []:
        tras_valuelist.append('\n')
    for tras_key1,tras_value1 in tras_value.iteritems():
        tras_valuelist.append(tras_value1)

for emis_key,emis_value in emis_pb.iteritems():
    if emis_valuelist != []:
        emis_valuelist.append('\n')
    for emis_key1,emis_value1 in emis_value.iteritems():
        emis_valuelist.append(emis_value1)

    





def isplit(iterable,splitters):
    return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]

print (start_valuelist)
tras_valuelist = isplit(tras_valuelist,('\n',))
emis_valuelist = isplit(emis_valuelist,('\n',))

print (start_valuelist)
print (tras_valuelist)
print (emis_valuelist)

print state
n_state = len(state)
print n_state
print observations
n_observations = len(observations)
print n_observations

model = hmm.MultinomialHMM(n_components=n_state)
model.startprob_ = np.array(start_valuelist)
model.transmat_ = np.array(tras_valuelist)
model.emissionprob_ = np.array(emis_valuelist)

# Predict the optimal sequence of internal hidden state
bob_says = np.array([[0, 2, 1, 1, 2, 0]]).T

model = model.fit(bob_says)

print model

print model.startprob_
print model.transmat_
print model.emissionprob_


logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")
print("Bob says:", ", ".join(map(lambda x: observations[x], [item for sublist in bob_says for item in sublist])))
print("Alice hears:", ", ".join(map(lambda x: states[x], alice_hears)))


'''


n_observations = len(observations)


transtates= []
transprob=[]
for key,value in trans_pb.iteritems():
    transtates.append(str(key))
    transprob.append(value)
print (transtates)
# print (transprob)
transprob = np.array([1,0,0,0,0,0,0.22,0.07,0.71])



model = hmm.MultinomialHMM(n_components=n_states, init_params="")
model.startprob_ = np.array(startprob)
print (model.startprob_)
model.transprob_ = np.array(transprob)
print (model.transprob_)

print (observations)
print (emissionprob)
'''