'''model1.py reads bams, trains model.'''


import pyranges
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import datetime

import numba
from numba import cuda
from numba import jit
import numpy as np
from timeit import default_timer as timer


begin_time = datetime.datetime.now()
print('Start: ', datetime.datetime.now())
#OUTPUT 2020-03-27 19:00:11.687429

dir_root = "/opt/student_homes/ieva.daukantas/Documents/genome/files3/vol1/run/"
#run_root = "ERR135/"
#run = "ERR135318/"
#dir = dir_root + run_root + run
##dir = "/opt/student_homes/ieva.daukantas/Documents/genome/files3/vol1/run/ERR135/ERR135318/"
#no = "7708_7#65"
#name = dir+no
#namebam = name+".bam"

'''1. save to pkl'''
def savetopkl(namebam, name):
    bam_df = pyranges.read_bam(namebam, sparse=False, as_df=True, mapq=0, required_flag=0, filter_flag=1540) 
    #print(bam_df)
    
    #print(bam_df.columns)
    #Index(['Chromosome', 'Start', 'End', 'Strand', 'Flag', 'QueryStart', 'QueryEnd', 'QuerySequence', 'Name', 'Cigar', 'Quality'], dtype='object')
    cols=['Chromosome', 'Start', 'End', 'Strand', 'Flag', 'QueryStart', 'QueryEnd', 'QuerySequence', 'Name', 'Cigar', 'Quality']
    
    #for col in cols:
    #    print(bam_df[col][0])

    bam_df.to_pickle(name+".pkl")



'''2. clean nan values'''
def clean(name):
    bam_df2 = pd.read_pickle(name+'.pkl')
    bam_df2.dropna(subset = ["Chromosome"], inplace=True)
    bam_df2.to_pickle(name+"_wonans.pkl")


'''3. sequences handling'''
def readwonans(name):
    bam_df3 = pd.read_pickle(name+"_wonans.pkl") 
    print('loaded from pickle _wonans')
    return bam_df3

#sequencestring = bam_df3['QuerySequence'][28711939]
#print(sequencestring)

def Kmers_funct(seq, size):
   return [seq[x:x+size].lower() for x in range(len(seq) - size + 1)]

#words = Kmers_funct(sequencestring, size=7)
#joined_sentence = ' '.join(words)
#print(words)
#print(joined_sentence)


##budas a
#@jit
#def intowords():
#    bam_df3['words'] = bam_df3.apply(lambda x: Kmers_funct(x['QuerySequence'], size=99), axis=1) #killed.

##budas b
#bam_df3["words"] = ""
#bam_df3['words'] = bam_df3['Start'] #cuda meta error
#bam_df4 = pd.DataFrame()
#bam_df4.colums() = ['words']

#size = 99 #it would take 14 days just to convert to words with size 6
#print(bam_df3)
#print('starting jit')

def intosentences(name, bam_df3, size):
    myfile = open(name +'.txt', 'w')
    for i in range(len(bam_df3['QuerySequence'])): #[0:3])):
        #print(i)
        sequencestring = bam_df3['QuerySequence'][i]
        words = Kmers_funct(sequencestring, size) #1 bamui 11 minučių
        #print(words)
        #bam_df3['QuerySequence'][i] = words #slows things down
        #bam_df4['words'][i] = words #slow

        joined_sentence = ' '.join(words)
        #print('joined sentence')
        
        myfile.write("%s\n" % joined_sentence)

        if i%3000000 == 0:
            #print('i=',i,'time:', datetime.datetime.now())
            now = datetime.datetime.now()
            diff = now - begin_time
            print(i, diff)
    myfile.close()



#intosentences(name, bam_df3, size)
#if __name__=="main":
#    start = timer()
#    #intowords()
#    intowords2()
#    print("with GPU: ", timer()-start)


#bam_df3 = bam_df3.drop('QuerySequence', axis=1)
#print(bam_df3)
#print(bam_df3.columns)
#bam_df3.to_pickle(name+"_words.pkl") #takes 1.5 min for 3 lines of bam when size=6


'''4. forming sets'''
def formsets(name, label):
    #print("reading txt.")
    bam_df4 = pd.read_csv(name+'.txt', sep="\n", header=None)
    bam_df4.columns = ["sentences"]
    bam_df4['label'] = label

    #print("saving pkl.")
    bam_df4.to_pickle(name + "_sets.pkl")
    #print("done.")
    print(bam_df4)

    bam_df5 = pd.read_pickle(name + "_sets.pkl")
    print(bam_df5)

#just testing #4:
#dir = "/opt/student_homes/ieva.daukantas/Documents/genome/files3/vol1/run/ERR178/ERR178285/"
#name = dir+ "8154_7#18"
#formsets(name, label=1)

##separate labels
#y_mouse = bam_df3.iloc[:, 0].values
#print(bam_df3)
#print(y_mouse)

'''5. choose bams for analysis'''
def choosebams(baminfo):
    data = pd.read_csv(baminfo, sep="\t", header=None)
    data.columns = ['Run_root', 'Run', 'bam', 'no', 'id']
    #print(data)
    return data

'''6. get dir and file names'''
def getnames(dir_root, row):
    run_root = row[0] + "/"
    run = row[1] + "/"
    dir = dir_root + run_root + run
    bam = row[2]
    no = row[3]
    name = dir+no
    namebam = name+".bam"
    id = row[4]
    return run_root, run, dir, no, name, namebam, id

'''7. join pickles'''
def joinpkl(names):
    print('concatinating pkls')
    df5_1113 = pd.concat([pd.read_pickle(name+"_sets.pkl") for name in names], axis=0)
    print('saving pkl')
    df5_1113.to_pickle("./df_1113_4.pkl")
    print('pkl saved.')
    #print("reading...")
    #df5 = pd.read_pickle("./df_1113_2.pkl")
    #print(df5)
    #example: df3 = pd.concat([name['sentences'] for name in names], axis=0)


def main():
    size = 99 #Kmers
    baminfo = "P11P13.txt"
    label = 1

    data = choosebams(baminfo) #5.
    i = 0

    names = []
    for index, row in data.iterrows():
        array = [row['Run_root'], row['Run'], row['bam'], row['no'], row['id']]
        run_root, run, dir, no, name, namebam, id = getnames(dir_root, array) #6.
        #print(run_root, '\n', run,  '\n', dir, '\n', no, '\n', name, '\n', namebam, '\n', id, '\n')
        
        #savetopkl(namebam, name) #1.
        #clean(name) #2.

        #bam_df3 = readwonans(name) #3.
        #intosentences(name, bam_df3, size)
        
        #formsets(name, label) #4.

        #i = i+1
        ##if i%10 == 0:
        #print("#### looping bam: ",i,"took time:", datetime.datetime.now()-begin_time)

        names.append(name)

    joinpkl(names) #7.

main()

print('End, ', datetime.datetime.now() - begin_time)




