# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 12:58:37 2020

@author: ablaa
"""
#To see parts that have to change for each experiment, go to SPECIFY...
#Exp # in excel and figure outputs have to be modified as well (Replace All)

# Import Stats
import statistics
import math
from scipy.stats import spearmanr
from scipy.stats import pearsonr
from scipy import stats

# Import Spacy

import spacy
from spacy.tokens import Doc
from spacy.vocab import Vocab

nlp = spacy.load("en_core_web_lg")

for word in nlp.Defaults.stop_words:
    lex = nlp.vocab[word]
    lex.is_stop = True

# Import MatplotLib
import matplotlib.pyplot as plt
import numpy as np

# Import Excel File
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

################
## CANON PORTION

#SPECIFY
num_gens=5
num_subs=25

#SPECIFY
df = pd.read_excel('StoryEvolution_AllData_Corrected.xlsx', sheet_name='Exp4_Canon')
 
print("Column headings:")
print(df.columns)

print(df['Gen_1'])

# Load in docs -- without Stop Words

Gen1_noSTOP=[]
Gen2_noSTOP=[]
Gen3_noSTOP=[]
Gen4_noSTOP=[]
Gen5_noSTOP=[]

text=[]
doc=[]

for d in range(num_subs):
    text=(df['Gen_1'][d])
    doc=nlp(text)
    noStop= [token.text for token in doc if not token.is_stop]
    doc = nlp(' '.join(noStop))
    Gen1_noSTOP.append(doc)
        
for d in range(num_subs):
    text=(df['Gen_2'][d])
    doc=nlp(text)
    noStop= [token.text for token in doc if not token.is_stop]
    doc = nlp(' '.join(noStop))
    Gen2_noSTOP.append(doc)
    
for d in range(num_subs):
    text=(df['Gen_3'][d])
    doc=nlp(text)
    noStop= [token.text for token in doc if not token.is_stop]
    doc = nlp(' '.join(noStop))
    Gen3_noSTOP.append(doc)
    
for d in range(num_subs):
    text=(df['Gen_4'][d])
    doc=nlp(text)
    noStop= [token.text for token in doc if not token.is_stop]
    doc = nlp(' '.join(noStop))
    Gen4_noSTOP.append(doc)
    
for d in range(num_subs):
    text=(df['Gen_5'][d])
    doc=nlp(text)
    noStop= [token.text for token in doc if not token.is_stop]
    doc = nlp(' '.join(noStop))
    Gen5_noSTOP.append(doc)

# Similarity of each subject, compared to next gen

#SPECIFY
story=open("sara_canon.txt").read()
doc0=nlp(story)
noStop= [token.text for token in doc0 if not token.is_stop]
doc0_noStop = nlp(' '.join(noStop))


Gen0=[]
Gen0_noSTOP=[]
for s in range(num_subs):
    Gen0.append(doc0)
    Gen0_noSTOP.append(doc0_noStop)
    

AllStories_NoStop={'Gen0':Gen0_noSTOP,'Gen1':Gen1_noSTOP,'Gen2':Gen2_noSTOP,'Gen3':Gen3_noSTOP,'Gen4':Gen4_noSTOP, 'Gen5':Gen5_noSTOP}


##Rank Order Correlation Measures -- CANON

## WHOLE NEW SECTION OF ANALYSES

#MEASURE ONE: Across Generations

#Forward

AllSubs_Ordered_NOSTOP={}
AllSubs_Compare_NOSTOP={}



AllSubs_OrderedMatch_NOSTOP={}
AllSubs_CompareMatch_NOSTOP={} 

AllSubs_OrderedOriginal={}
AllSubs_CompareOriginal={}
AllSubs_HighestOriginal={}
AllSubs_LowestOriginal={}



doclistNS=[]

for x in range(num_subs): 
    SubName='Sub{}'.format(x+1)
    #doclist=[AllStories['Gen0'][x], AllStories['Gen1'][x], AllStories['Gen2'][x], AllStories['Gen3'][x], AllStories['Gen4'][x], AllStories['Gen5'][x]]
    doclistNS=[AllStories_NoStop['Gen0'][x], AllStories_NoStop['Gen1'][x], AllStories_NoStop['Gen2'][x], AllStories_NoStop['Gen3'][x], AllStories_NoStop['Gen4'][x], AllStories_NoStop['Gen5'][x]]
    num_doc=len(doclistNS)
    
    sentences=[]
    num_sents=[]
    
    # Parse sentences for all docs
    for d in doclistNS:
        sentences.append(list(d.sents))
        num_sents.append(len(list(d.sents)))
    
    # Compare semantic similarities of sentences
    #Comparing Forward -- NO STOP WORDS
    Ordered_NOSTOP={}
    Compare_NOSTOP={} 
       
    for g in range(num_doc-1):
        similarity_sent=[[] for i in range(num_sents[g])]
        max_sim=[]
        for b in range(num_sents[g]): #for sentence in base doc
            docA=sentences[g][b]
            noStop= [token.text for token in docA if not token.is_stop]
            docA = nlp(' '.join(noStop))
            for c in range(num_sents[g+1]): #for sentence in compared doc
                docB=sentences[g+1][c] 
                noStop= [token.text for token in docB if not token.is_stop]
                docB = nlp(' '.join(noStop))
                similarity_sent[b].append(docA.similarity(docB))
            max_sim.append(1+similarity_sent[b].index(max(similarity_sent[b])))
        
        CompName='Gen{}vnext'.format(g)
        Ordered_NOSTOP[CompName]=max_sim #Find sentence most similar in each generation
        Compare_NOSTOP[CompName]=similarity_sent #Compare: Each gen, by each first gen sentence, by next gen sentence


    AllSubs_Ordered_NOSTOP[SubName]=Ordered_NOSTOP
    AllSubs_Compare_NOSTOP[SubName]=Ordered_NOSTOP 

    # Based on NO STOP Versions
OriGen=[[] for i in range(num_subs)]
FirstGen=[[] for i in range(num_subs)]
SecGen=[[] for i in range(num_subs)]
ThirGen=[[] for i in range(num_subs)]
FourGen=[[] for i in range(num_subs)]
OriCom=[[] for i in range(num_subs)]
FirstCom=[[] for i in range(num_subs)]
SecCom=[[] for i in range(num_subs)]
ThirCom=[[] for i in range(num_subs)]
FourCom=[[] for i in range(num_subs)]
OriCorr=[]
FirstCorr=[]
SecCorr=[]
ThirCorr=[]
FourCorr=[]
CorrTemp=[]

for s in range(num_subs):
    OriGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen0vnext']
    OriCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen0vnext']))))
    CorrTemp=spearmanr(OriCom[s],OriGen[s])
    OriCorr.append(CorrTemp[0])
    FirstGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vnext']
    FirstCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vnext']))))
    CorrTemp=spearmanr(FirstCom[s],FirstGen[s])
    FirstCorr.append(CorrTemp[0])
    SecGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vnext']
    SecCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vnext']))))
    CorrTemp=spearmanr(SecCom[s],SecGen[s])
    SecCorr.append(CorrTemp[0])
    ThirGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vnext']
    ThirCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vnext']))))
    CorrTemp=spearmanr(ThirCom[s],ThirGen[s])
    ThirCorr.append(CorrTemp[0])
    FourGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vnext']
    FourCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vnext']))))
    CorrTemp=spearmanr(FourCom[s],FourGen[s])
    FourCorr.append(CorrTemp[0])
    
Canon_Forwards=OriCorr + FirstCorr + SecCorr + ThirCorr + FourCorr
Canon_AllCorr=[OriCorr, FirstCorr, SecCorr, ThirCorr, FourCorr]
Can_df_AllCorr = pd.DataFrame(Canon_AllCorr)
Can_df_AllCorr = pd.DataFrame.transpose(Can_df_AllCorr)
print(Can_df_AllCorr)
Can_df_AllCorr.to_excel("Exp4_Canon_OrderComparisonAcrossSubs_Spacy.xlsx",sheet_name='SentCorrelations')

Can_MeanCorr_Across=[statistics.mean(OriCorr), statistics.mean(FirstCorr),statistics.mean(SecCorr),statistics.mean(ThirCorr),statistics.mean(FourCorr)]
Can_OriCorr_Across=OriCorr
Can_FirstCorr_Across=FirstCorr
Can_SecCorr_Across=SecCorr
Can_ThirCorr_Across=ThirCorr
Can_FourCorr_Across=FourCorr

Can_StDError_Across=[stats.sem(OriCorr), stats.sem(FirstCorr),stats.sem(SecCorr),stats.sem(ThirCorr),stats.sem(FourCorr)]
Can_StDev_Across=[stats.tstd(OriCorr), stats.tstd(FirstCorr),stats.tstd(SecCorr),stats.tstd(ThirCorr),stats.tstd(FourCorr)]

objects = ('Original to Gen1', 'Gen1 to Gen2', 'Gen2 to Gen3', 'Gen3 to Gen4', 'Gen4 to Gen5')
y_pos = np.arange(len(objects))


plt.errorbar(y_pos*5, Can_MeanCorr_Across, yerr=Can_StDError_Across)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Canon - Average Correlation Across Generations (Spacy)')
plt.savefig('Exp4_Can_Spacy_CorrOrder_Next')
plt.clf()


#Backwards

AllSubs_Ordered_NOSTOP={}
AllSubs_Compare_NOSTOP={}



AllSubs_OrderedMatch_NOSTOP={}
AllSubs_CompareMatch_NOSTOP={} 

AllSubs_OrderedOriginal={}
AllSubs_CompareOriginal={}
AllSubs_HighestOriginal={}
AllSubs_LowestOriginal={}



doclistNS=[]

for x in range(num_subs): 
    SubName='Sub{}'.format(x+1)
    #doclist=[AllStories['Gen0'][x], AllStories['Gen1'][x], AllStories['Gen2'][x], AllStories['Gen3'][x], AllStories['Gen4'][x], AllStories['Gen5'][x]]
    doclistNS=[AllStories_NoStop['Gen0'][x], AllStories_NoStop['Gen1'][x], AllStories_NoStop['Gen2'][x], AllStories_NoStop['Gen3'][x], AllStories_NoStop['Gen4'][x], AllStories_NoStop['Gen5'][x]]
    num_doc=len(doclistNS)
    
    sentences=[]
    num_sents=[]
    
    # Parse sentences for all docs
    for d in doclistNS:
        sentences.append(list(d.sents))
        num_sents.append(len(list(d.sents)))
    
    # Compare semantic similarities of sentences
    #Comparing Forward -- NO STOP WORDS
    Ordered_NOSTOP={}
    Compare_NOSTOP={} 
       
    for g in range(num_doc-1):
        similarity_sent=[[] for i in range(num_sents[g+1])]
        max_sim=[]
        for b in range(num_sents[g+1]): #for sentence in base doc
            docA=sentences[g+1][b]
            noStop= [token.text for token in docA if not token.is_stop]
            docA = nlp(' '.join(noStop))
            for c in range(num_sents[g]): #for sentence in compared doc
                docB=sentences[g][c] 
                noStop= [token.text for token in docB if not token.is_stop]
                docB = nlp(' '.join(noStop))
                similarity_sent[b].append(docA.similarity(docB))
            max_sim.append(1+similarity_sent[b].index(max(similarity_sent[b])))
        
        CompName='Gen{}vprevious'.format(g+1)
        Ordered_NOSTOP[CompName]=max_sim #Find sentence most similar in each generation
        Compare_NOSTOP[CompName]=similarity_sent #Compare: Each gen, by each first gen sentence, by next gen sentence


    AllSubs_Ordered_NOSTOP[SubName]=Ordered_NOSTOP
    AllSubs_Compare_NOSTOP[SubName]=Ordered_NOSTOP 

    # Based on NO STOP Versions
OriGen=[[] for i in range(num_subs)]
FirstGen=[[] for i in range(num_subs)]
SecGen=[[] for i in range(num_subs)]
ThirGen=[[] for i in range(num_subs)]
FourGen=[[] for i in range(num_subs)]
OriCom=[[] for i in range(num_subs)]
FirstCom=[[] for i in range(num_subs)]
SecCom=[[] for i in range(num_subs)]
ThirCom=[[] for i in range(num_subs)]
FourCom=[[] for i in range(num_subs)]
OriCorr=[]
FirstCorr=[]
SecCorr=[]
ThirCorr=[]
FourCorr=[]
CorrTemp=[]

for s in range(num_subs):
    OriGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vprevious']
    OriCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vprevious']))))
    CorrTemp=spearmanr(OriCom[s],OriGen[s])
    OriCorr.append(CorrTemp[0])
    FirstGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vprevious']
    FirstCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vprevious']))))
    CorrTemp=spearmanr(FirstCom[s],FirstGen[s])
    FirstCorr.append(CorrTemp[0])
    SecGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vprevious']
    SecCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vprevious']))))
    CorrTemp=spearmanr(SecCom[s],SecGen[s])
    SecCorr.append(CorrTemp[0])
    ThirGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vprevious']
    ThirCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vprevious']))))
    CorrTemp=spearmanr(ThirCom[s],ThirGen[s])
    ThirCorr.append(CorrTemp[0])
    FourGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5vprevious']
    FourCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5vprevious']))))
    CorrTemp=spearmanr(FourCom[s],FourGen[s])
    FourCorr.append(CorrTemp[0])
    
Canon_Backwards=OriCorr + FirstCorr + SecCorr + ThirCorr + FourCorr
Canon_AllCorr=[OriCorr, FirstCorr, SecCorr, ThirCorr, FourCorr]
Can_df_AllCorr = pd.DataFrame(Canon_AllCorr)
Can_df_AllCorr = pd.DataFrame.transpose(Can_df_AllCorr)
print(Can_df_AllCorr)
Can_df_AllCorr.to_excel("Exp4_Canon_OrderComparisonAcrossSubs_Spacy_Backwards.xlsx",sheet_name='SentCorrelations')

Can_MeanCorr_Across_Back=[statistics.mean(OriCorr), statistics.mean(FirstCorr),statistics.mean(SecCorr),statistics.mean(ThirCorr),statistics.mean(FourCorr)]
Can_OriCorr_Across_Back=OriCorr
Can_FirstCorr_Across_Back=FirstCorr
Can_SecCorr_Across_Back=SecCorr
Can_ThirCorr_Across_Back=ThirCorr
Can_FourCorr_Across_Back=FourCorr

Can_StDError_Across_Back=[stats.sem(OriCorr), stats.sem(FirstCorr),stats.sem(SecCorr),stats.sem(ThirCorr),stats.sem(FourCorr)]
Can_StDev_Across_Back=[stats.tstd(OriCorr), stats.tstd(FirstCorr),stats.tstd(SecCorr),stats.tstd(ThirCorr),stats.tstd(FourCorr)]

objects = ('Original to Gen1', 'Gen1 to Gen2', 'Gen2 to Gen3', 'Gen3 to Gen4', 'Gen4 to Gen5')
y_pos = np.arange(len(objects))


plt.errorbar(y_pos*5, Can_MeanCorr_Across_Back, yerr=Can_StDError_Across_Back)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Canon - Average Correlation Across Generations (Spacy - Backwards)')
plt.savefig('Exp4_Can_Spacy_CorrOrder_Backwards')
plt.clf()


##GET CORRELATION OF TWO MEASURES

Canon_Correlation_ForBack=pearsonr(Canon_Forwards, Canon_Backwards)
print(Canon_Correlation_ForBack)

#######

#MEASURE TWO: TO LAST

#With Earlier Gens as Reference

AllSubs_Ordered_NOSTOP={}
AllSubs_Compare_NOSTOP={}



AllSubs_OrderedMatch_NOSTOP={}
AllSubs_CompareMatch_NOSTOP={} 

AllSubs_OrderedOriginal={}
AllSubs_CompareOriginal={}
AllSubs_HighestOriginal={}
AllSubs_LowestOriginal={}



doclistNS=[]

for x in range(num_subs): 
    SubName='Sub{}'.format(x+1)
    #doclist=[AllStories['Gen0'][x], AllStories['Gen1'][x], AllStories['Gen2'][x], AllStories['Gen3'][x], AllStories['Gen4'][x], AllStories['Gen5'][x]]
    doclistNS=[AllStories_NoStop['Gen0'][x], AllStories_NoStop['Gen1'][x], AllStories_NoStop['Gen2'][x], AllStories_NoStop['Gen3'][x], AllStories_NoStop['Gen4'][x], AllStories_NoStop['Gen5'][x]]
    num_doc=len(doclistNS)
    
    sentences=[]
    num_sents=[]
    
    # Parse sentences for all docs
    for d in doclistNS:
        sentences.append(list(d.sents))
        num_sents.append(len(list(d.sents)))
    
    # Compare semantic similarities of sentences
    #Comparing Forward -- NO STOP WORDS
    Ordered_NOSTOP={}
    Compare_NOSTOP={} 
       
    for g in range(num_doc-1):
        similarity_sent=[[] for i in range(num_sents[g])]
        max_sim=[]
        for b in range(num_sents[g]): #for sentence in base doc
            docA=sentences[g][b]
            noStop= [token.text for token in docA if not token.is_stop]
            docA = nlp(' '.join(noStop))
            for c in range(num_sents[5]): #for sentence in compared doc
                docB=sentences[5][c] 
                noStop= [token.text for token in docB if not token.is_stop]
                docB = nlp(' '.join(noStop))
                similarity_sent[b].append(docA.similarity(docB))
            max_sim.append(1+similarity_sent[b].index(max(similarity_sent[b])))
        
        CompName='Gen{}vlast'.format(g)
        Ordered_NOSTOP[CompName]=max_sim #Find sentence most similar in each generation
        Compare_NOSTOP[CompName]=similarity_sent #Compare: Each gen, by each first gen sentence, by next gen sentence


    AllSubs_Ordered_NOSTOP[SubName]=Ordered_NOSTOP
    AllSubs_Compare_NOSTOP[SubName]=Ordered_NOSTOP 

    # Based on NO STOP Versions
OriGen=[[] for i in range(num_subs)]
FirstGen=[[] for i in range(num_subs)]
SecGen=[[] for i in range(num_subs)]
ThirGen=[[] for i in range(num_subs)]
FourGen=[[] for i in range(num_subs)]
OriCom=[[] for i in range(num_subs)]
FirstCom=[[] for i in range(num_subs)]
SecCom=[[] for i in range(num_subs)]
ThirCom=[[] for i in range(num_subs)]
FourCom=[[] for i in range(num_subs)]
OriCorr=[]
FirstCorr=[]
SecCorr=[]
ThirCorr=[]
FourCorr=[]
CorrTemp=[]

for s in range(num_subs):
    OriGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen0vlast']
    OriCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen0vlast']))))
    CorrTemp=spearmanr(OriCom[s],OriGen[s])
    OriCorr.append(CorrTemp[0])
    FirstGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vlast']
    FirstCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vlast']))))
    CorrTemp=spearmanr(FirstCom[s],FirstGen[s])
    FirstCorr.append(CorrTemp[0])
    SecGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vlast']
    SecCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vlast']))))
    CorrTemp=spearmanr(SecCom[s],SecGen[s])
    SecCorr.append(CorrTemp[0])
    ThirGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vlast']
    ThirCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vlast']))))
    CorrTemp=spearmanr(ThirCom[s],ThirGen[s])
    ThirCorr.append(CorrTemp[0])
    FourGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vlast']
    FourCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vlast']))))
    CorrTemp=spearmanr(FourCom[s],FourGen[s])
    FourCorr.append(CorrTemp[0])
    

Can_MeanCorr_Last=[statistics.mean(OriCorr), statistics.mean(FirstCorr),statistics.mean(SecCorr),statistics.mean(ThirCorr),statistics.mean(FourCorr)]
Can_OriCorr_Last=OriCorr
Can_FirstCorr_Last=FirstCorr
Can_SecCorr_Last=SecCorr
Can_ThirCorr_Last=ThirCorr
Can_FourCorr_Last=FourCorr
Can_Error_Last=[stats.sem(OriCorr), stats.sem(FirstCorr),stats.sem(SecCorr),stats.sem(ThirCorr),stats.sem(FourCorr)]
Can_StDev_Last=[stats.tstd(OriCorr), stats.tstd(FirstCorr),stats.tstd(SecCorr),stats.tstd(ThirCorr),stats.tstd(FourCorr)]

objects = ('Original to Gen5', 'Gen1 to Gen5', 'Gen2 to Gen5', 'Gen3 to Gen5', 'Gen4 to Gen5')
y_pos = np.arange(len(objects))


plt.errorbar(y_pos*5, Can_MeanCorr_Last, yerr=Can_Error_Last)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Canon - Average Correlation to Last Generation (Spacy')
plt.savefig('Exp4_Can_Spacy_CorrOrdertoLast')
plt.clf()


Canon_Last=Can_OriCorr_Last + Can_FirstCorr_Last + Can_SecCorr_Last + Can_ThirCorr_Last + Can_FourCorr_Last
Canon_AllCorr_Last=[Can_OriCorr_Last, Can_FirstCorr_Last, Can_SecCorr_Last, Can_ThirCorr_Last, Can_FourCorr_Last]
Can_df_AllCorr_Last = pd.DataFrame(Canon_AllCorr_Last)
Can_df_AllCorr_Last = pd.DataFrame.transpose(Can_df_AllCorr_Last)
print(Can_df_AllCorr_Last)
Can_df_AllCorr_Last.to_excel("Exp4_Canon_OrderComparisontoLast_Spacy.xlsx",sheet_name='SentCorrelations')


#With last Gen as Reference

AllSubs_Ordered_NOSTOP={}
AllSubs_Compare_NOSTOP={}



AllSubs_OrderedMatch_NOSTOP={}
AllSubs_CompareMatch_NOSTOP={} 

AllSubs_OrderedOriginal={}
AllSubs_CompareOriginal={}
AllSubs_HighestOriginal={}
AllSubs_LowestOriginal={}



doclistNS=[]

for x in range(num_subs): 
    SubName='Sub{}'.format(x+1)
    #doclist=[AllStories['Gen0'][x], AllStories['Gen1'][x], AllStories['Gen2'][x], AllStories['Gen3'][x], AllStories['Gen4'][x], AllStories['Gen5'][x]]
    doclistNS=[AllStories_NoStop['Gen0'][x], AllStories_NoStop['Gen1'][x], AllStories_NoStop['Gen2'][x], AllStories_NoStop['Gen3'][x], AllStories_NoStop['Gen4'][x], AllStories_NoStop['Gen5'][x]]
    num_doc=len(doclistNS)
    
    sentences=[]
    num_sents=[]
    
    # Parse sentences for all docs
    for d in doclistNS:
        sentences.append(list(d.sents))
        num_sents.append(len(list(d.sents)))
    
    # Compare semantic similarities of sentences
    #Comparing Forward -- NO STOP WORDS
    Ordered_NOSTOP={}
    Compare_NOSTOP={} 
       
    for g in range(num_doc-1):
        similarity_sent=[[] for i in range(num_sents[5])]
        max_sim=[]
        for b in range(num_sents[5]): #for sentence in base doc
            docA=sentences[5][b]
            noStop= [token.text for token in docA if not token.is_stop]
            docA = nlp(' '.join(noStop))
            for c in range(num_sents[g]): #for sentence in compared doc
                docB=sentences[g][c] 
                noStop= [token.text for token in docB if not token.is_stop]
                docB = nlp(' '.join(noStop))
                similarity_sent[b].append(docA.similarity(docB))
            max_sim.append(1+similarity_sent[b].index(max(similarity_sent[b])))
        
        CompName='Gen5v{}'.format(g)
        Ordered_NOSTOP[CompName]=max_sim #Find sentence most similar in each generation
        Compare_NOSTOP[CompName]=similarity_sent #Compare: Each gen, by each first gen sentence, by next gen sentence


    AllSubs_Ordered_NOSTOP[SubName]=Ordered_NOSTOP
    AllSubs_Compare_NOSTOP[SubName]=Ordered_NOSTOP 

    # Based on NO STOP Versions
OriGen=[[] for i in range(num_subs)]
FirstGen=[[] for i in range(num_subs)]
SecGen=[[] for i in range(num_subs)]
ThirGen=[[] for i in range(num_subs)]
FourGen=[[] for i in range(num_subs)]
OriCom=[[] for i in range(num_subs)]
FirstCom=[[] for i in range(num_subs)]
SecCom=[[] for i in range(num_subs)]
ThirCom=[[] for i in range(num_subs)]
FourCom=[[] for i in range(num_subs)]
OriCorr=[]
FirstCorr=[]
SecCorr=[]
ThirCorr=[]
FourCorr=[]
CorrTemp=[]

for s in range(num_subs):
    OriGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v0']
    OriCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v0']))))
    CorrTemp=spearmanr(OriCom[s],OriGen[s])
    OriCorr.append(CorrTemp[0])
    FirstGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v1']
    FirstCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v1']))))
    CorrTemp=spearmanr(FirstCom[s],FirstGen[s])
    FirstCorr.append(CorrTemp[0])
    SecGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v2']
    SecCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v2']))))
    CorrTemp=spearmanr(SecCom[s],SecGen[s])
    SecCorr.append(CorrTemp[0])
    ThirGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v3']
    ThirCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v3']))))
    CorrTemp=spearmanr(ThirCom[s],ThirGen[s])
    ThirCorr.append(CorrTemp[0])
    FourGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v4']
    FourCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v4']))))
    CorrTemp=spearmanr(FourCom[s],FourGen[s])
    FourCorr.append(CorrTemp[0])
    

Can_MeanCorr_LastRef=[statistics.mean(OriCorr), statistics.mean(FirstCorr),statistics.mean(SecCorr),statistics.mean(ThirCorr),statistics.mean(FourCorr)]
Can_OriCorr_LastRef=OriCorr
Can_FirstCorr_LastRef=FirstCorr
Can_SecCorr_LastRef=SecCorr
Can_ThirCorr_LastRef=ThirCorr
Can_FourCorr_LastRef=FourCorr
Can_Error_LastRef=[stats.sem(OriCorr), stats.sem(FirstCorr),stats.sem(SecCorr),stats.sem(ThirCorr),stats.sem(FourCorr)]
Can_StDev_LastRef=[stats.tstd(OriCorr), stats.tstd(FirstCorr),stats.tstd(SecCorr),stats.tstd(ThirCorr),stats.tstd(FourCorr)]

objects = ('Gen5 to Original', 'Gen5 to Gen1', 'Gen5 to Gen2', 'Gen5 to Gen3', 'Gen5 to Gen4')
y_pos = np.arange(len(objects))


plt.errorbar(y_pos*5, Can_MeanCorr_LastRef, yerr=Can_Error_LastRef)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Canon - Average Correlation to Last Generation (Spacy)_RefLast')
plt.savefig('Exp4_Can_Spacy_CorrOrdertoLast_RefLast')
plt.clf()


Canon_RefLast=Can_OriCorr_LastRef + Can_FirstCorr_LastRef + Can_SecCorr_LastRef + Can_ThirCorr_LastRef + Can_FourCorr_LastRef
Canon_AllCorr_Last=[Can_OriCorr_LastRef, Can_FirstCorr_LastRef, Can_SecCorr_LastRef, Can_ThirCorr_LastRef, Can_FourCorr_LastRef]
Can_df_AllCorr_Last = pd.DataFrame(Canon_AllCorr_Last)
Can_df_AllCorr_Last = pd.DataFrame.transpose(Can_df_AllCorr_Last)
print(Can_df_AllCorr_Last)
Can_df_AllCorr_Last.to_excel("Exp4_Canon_OrderComparisontoLast_Spacy_RefLast.xlsx",sheet_name='SentCorrelations')

##GET CORRELATION OF TWO MEASURES

Canon_Correlation_RefLast=pearsonr(Canon_Last, Canon_RefLast)
print(Canon_Correlation_RefLast)

##MEASURE THREE: Comparison to INITIAL
## With Initial Story as Reference

AllSubs_Ordered_NOSTOP={}
AllSubs_Compare_NOSTOP={}



AllSubs_OrderedMatch_NOSTOP={}
AllSubs_CompareMatch_NOSTOP={} 

AllSubs_OrderedOriginal={}
AllSubs_CompareOriginal={}
AllSubs_HighestOriginal={}
AllSubs_LowestOriginal={}



doclistNS=[]

for x in range(num_subs): 
    SubName='Sub{}'.format(x+1)
    doclistNS=[AllStories_NoStop['Gen0'][x], AllStories_NoStop['Gen1'][x], AllStories_NoStop['Gen2'][x], AllStories_NoStop['Gen3'][x], AllStories_NoStop['Gen4'][x], AllStories_NoStop['Gen5'][x]]
    num_doc=len(doclistNS)
    
    sentences=[]
    num_sents=[]
    
    # Parse sentences for all docs
    for d in doclistNS:
        sentences.append(list(d.sents))
        num_sents.append(len(list(d.sents)))
    
    # Compare semantic similarities of sentences
    #Comparing Forward -- NO STOP WORDS
    Ordered_NOSTOP={}
    Compare_NOSTOP={} 
       
    for g in range(1,num_doc):
        similarity_sent=[[] for i in range(num_sents[0])]
        max_sim=[]
        for b in range(num_sents[0]): #for sentence in base doc
            docA=sentences[0][b]
            noStop= [token.text for token in docA if not token.is_stop]
            docA = nlp(' '.join(noStop))
            for c in range(num_sents[g]): #for sentence in compared doc
                docB=sentences[g][c] 
                noStop= [token.text for token in docB if not token.is_stop]
                docB = nlp(' '.join(noStop))
                similarity_sent[b].append(docA.similarity(docB))
            max_sim.append(1+similarity_sent[b].index(max(similarity_sent[b])))
        
        CompName='FirstvGen{}'.format(g)
        Ordered_NOSTOP[CompName]=max_sim #Find sentence most similar in each generation
        Compare_NOSTOP[CompName]=similarity_sent #Compare: Each gen, by each first gen sentence, by next gen sentence


    AllSubs_Ordered_NOSTOP[SubName]=Ordered_NOSTOP
    AllSubs_Compare_NOSTOP[SubName]=Ordered_NOSTOP 

    # Based on NO STOP Versions
OriGen=[[] for i in range(num_subs)]
FirstGen=[[] for i in range(num_subs)]
SecGen=[[] for i in range(num_subs)]
ThirGen=[[] for i in range(num_subs)]
FourGen=[[] for i in range(num_subs)]
OriCom=[[] for i in range(num_subs)]
FirstCom=[[] for i in range(num_subs)]
SecCom=[[] for i in range(num_subs)]
ThirCom=[[] for i in range(num_subs)]
FourCom=[[] for i in range(num_subs)]
OriCorr=[]
FirstCorr=[]
SecCorr=[]
ThirCorr=[]
FourCorr=[]
CorrTemp=[]

for s in range(num_subs):
    OriGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen1']
    OriCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen1']))))
    CorrTemp=spearmanr(OriCom[s],OriGen[s])
    OriCorr.append(CorrTemp[0])
    FirstGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen2']
    FirstCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen2']))))
    CorrTemp=spearmanr(FirstCom[s],FirstGen[s])
    FirstCorr.append(CorrTemp[0])
    SecGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen3']
    SecCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen3']))))
    CorrTemp=spearmanr(SecCom[s],SecGen[s])
    SecCorr.append(CorrTemp[0])
    ThirGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen4']
    ThirCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen4']))))
    CorrTemp=spearmanr(ThirCom[s],ThirGen[s])
    ThirCorr.append(CorrTemp[0])
    FourGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen5']
    FourCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen5']))))
    CorrTemp=spearmanr(FourCom[s],FourGen[s])
    FourCorr.append(CorrTemp[0])
    

Can_MeanCorr_First=[statistics.mean(OriCorr), statistics.mean(FirstCorr),statistics.mean(SecCorr),statistics.mean(ThirCorr),statistics.mean(FourCorr)]
Can_OriCorr_First=OriCorr
Can_FirstCorr_First=FirstCorr
Can_SecCorr_First=SecCorr
Can_ThirCorr_First=ThirCorr
Can_FourCorr_First=FourCorr
Can_Error_First=[stats.sem(OriCorr), stats.sem(FirstCorr),stats.sem(SecCorr),stats.sem(ThirCorr),stats.sem(FourCorr)]
Can_StDev_First=[stats.tstd(OriCorr), stats.tstd(FirstCorr),stats.tstd(SecCorr),stats.tstd(ThirCorr),stats.tstd(FourCorr)]

objects = ('Original to Gen1', 'Original to Gen2', 'Original to Gen3', 'Original to Gen4', 'Original to Gen5')
y_pos = np.arange(len(objects))


plt.errorbar(y_pos*5, Can_MeanCorr_First, yerr=Can_Error_First)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Canon - Average Correlation to Initial (Spacy)')
plt.savefig('Exp4_Can_Spacy_CorrOrdertoFirst_InitialReference')
plt.clf()


Canon_First=Can_OriCorr_First + Can_FirstCorr_First + Can_SecCorr_First + Can_ThirCorr_First + Can_FourCorr_First
Canon_AllCorr_First=[Can_OriCorr_First, Can_FirstCorr_First, Can_SecCorr_First, Can_ThirCorr_First, Can_FourCorr_First]
Can_df_AllCorr_First = pd.DataFrame(Canon_AllCorr_First)
Can_df_AllCorr_First = pd.DataFrame.transpose(Can_df_AllCorr_First)
print(Can_df_AllCorr_First)
Can_df_AllCorr_First.to_excel("Exp4_Canon_OrderComparisontoFirst_Spacy.xlsx",sheet_name='SentCorrelations')


## With other as Reference

AllSubs_Ordered_NOSTOP={}
AllSubs_Compare_NOSTOP={}



AllSubs_OrderedMatch_NOSTOP={}
AllSubs_CompareMatch_NOSTOP={} 

AllSubs_OrderedOriginal={}
AllSubs_CompareOriginal={}
AllSubs_HighestOriginal={}
AllSubs_LowestOriginal={}



doclistNS=[]

for x in range(num_subs): 
    SubName='Sub{}'.format(x+1)
    doclistNS=[AllStories_NoStop['Gen0'][x], AllStories_NoStop['Gen1'][x], AllStories_NoStop['Gen2'][x], AllStories_NoStop['Gen3'][x], AllStories_NoStop['Gen4'][x], AllStories_NoStop['Gen5'][x]]
    num_doc=len(doclistNS)
    
    sentences=[]
    num_sents=[]
    
    # Parse sentences for all docs
    for d in doclistNS:
        sentences.append(list(d.sents))
        num_sents.append(len(list(d.sents)))
    
    # Compare semantic similarities of sentences
    Ordered_NOSTOP={}
    Compare_NOSTOP={} 
       
    for g in range(1,num_doc):
        similarity_sent=[[] for i in range(num_sents[g])]
        max_sim=[]
        for b in range(num_sents[g]): #for sentence in base doc
            docA=sentences[g][b]
            noStop= [token.text for token in docA if not token.is_stop]
            docA = nlp(' '.join(noStop))
            for c in range(num_sents[0]): #for sentence in compared doc
                docB=sentences[0][c] 
                noStop= [token.text for token in docB if not token.is_stop]
                docB = nlp(' '.join(noStop))
                similarity_sent[b].append(docA.similarity(docB))
            max_sim.append(1+similarity_sent[b].index(max(similarity_sent[b])))
        
        CompName='Gen{}vfirst'.format(g)
        Ordered_NOSTOP[CompName]=max_sim #Find sentence most similar in each generation
        Compare_NOSTOP[CompName]=similarity_sent #Compare: Each gen, by each first gen sentence, by next gen sentence


    AllSubs_Ordered_NOSTOP[SubName]=Ordered_NOSTOP
    AllSubs_Compare_NOSTOP[SubName]=Ordered_NOSTOP 

    # Based on NO STOP Versions
OriGen=[[] for i in range(num_subs)]
FirstGen=[[] for i in range(num_subs)]
SecGen=[[] for i in range(num_subs)]
ThirGen=[[] for i in range(num_subs)]
FourGen=[[] for i in range(num_subs)]
OriCom=[[] for i in range(num_subs)]
FirstCom=[[] for i in range(num_subs)]
SecCom=[[] for i in range(num_subs)]
ThirCom=[[] for i in range(num_subs)]
FourCom=[[] for i in range(num_subs)]
OriCorr=[]
FirstCorr=[]
SecCorr=[]
ThirCorr=[]
FourCorr=[]
CorrTemp=[]

for s in range(num_subs):
    OriGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vfirst']
    OriCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vfirst']))))
    CorrTemp=spearmanr(OriCom[s],OriGen[s])
    OriCorr.append(CorrTemp[0])
    FirstGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vfirst']
    FirstCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vfirst']))))
    CorrTemp=spearmanr(FirstCom[s],FirstGen[s])
    FirstCorr.append(CorrTemp[0])
    SecGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vfirst']
    SecCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vfirst']))))
    CorrTemp=spearmanr(SecCom[s],SecGen[s])
    SecCorr.append(CorrTemp[0])
    ThirGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vfirst']
    ThirCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vfirst']))))
    CorrTemp=spearmanr(ThirCom[s],ThirGen[s])
    ThirCorr.append(CorrTemp[0])
    FourGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5vfirst']
    FourCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5vfirst']))))
    CorrTemp=spearmanr(FourCom[s],FourGen[s])
    FourCorr.append(CorrTemp[0])
    

Can_MeanCorr_First_Back=[statistics.mean(OriCorr), statistics.mean(FirstCorr),statistics.mean(SecCorr),statistics.mean(ThirCorr),statistics.mean(FourCorr)]
Can_OriCorr_First_Back=OriCorr
Can_FirstCorr_First_Back=FirstCorr
Can_SecCorr_First_Back=SecCorr
Can_ThirCorr_First_Back=ThirCorr
Can_FourCorr_First_Back=FourCorr
Can_Error_First_Back=[stats.sem(OriCorr), stats.sem(FirstCorr),stats.sem(SecCorr),stats.sem(ThirCorr),stats.sem(FourCorr)]
Can_StDev_First_Back=[stats.tstd(OriCorr), stats.tstd(FirstCorr),stats.tstd(SecCorr),stats.tstd(ThirCorr),stats.tstd(FourCorr)]

objects = ('Gen1 to Original', 'Gen2 to Original', 'Gen3 to Original', 'Gen4 to Original', 'Gen5 to Original')
y_pos = np.arange(len(objects))


plt.errorbar(y_pos*5, Can_MeanCorr_First_Back, yerr=Can_Error_First_Back)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Canon - Average Correlation to Initial (Spacy)- OtherRef')
plt.savefig('Exp4_Can_Spacy_CorrOrdertoFirst_OtherReference')
plt.clf()



Canon_First_RefOther=Can_OriCorr_First_Back + Can_FirstCorr_First_Back + Can_SecCorr_First_Back + Can_ThirCorr_First_Back + Can_FourCorr_First_Back
Canon_AllCorr_First=[Can_OriCorr_First_Back, Can_FirstCorr_First_Back, Can_SecCorr_First_Back, Can_ThirCorr_First_Back, Can_FourCorr_First_Back]
Can_df_AllCorr_First = pd.DataFrame(Canon_AllCorr_First)
Can_df_AllCorr_First = pd.DataFrame.transpose(Can_df_AllCorr_First)
print(Can_df_AllCorr_First)
Can_df_AllCorr_First.to_excel("Exp4_Canon_OrderComparisontoFirst_Spacy_RefOther.xlsx",sheet_name='SentCorrelations')

##GET CORRELATION OF TWO MEASURES

Canon_Correlation_RefFirst=pearsonr(Canon_First, Canon_First_RefOther)
print(Canon_Correlation_RefFirst)


#################

### NONCANON PORTION
#SPECIFY...
num_gens=5
num_subs=25

#SPECIFY...
df = pd.read_excel('StoryEvolution_AllData_Corrected.xlsx', sheet_name='Exp4_NonCanon')
 
 
print("Column headings:")
print(df.columns)

print(df['Gen_1'])


# Load in docs -- without Stop Words

Gen1_noSTOP=[]
Gen2_noSTOP=[]
Gen3_noSTOP=[]
Gen4_noSTOP=[]
Gen5_noSTOP=[]

text=[]
doc=[]

for d in range(num_subs):
    text=(df['Gen_1'][d])
    doc=nlp(text)
    noStop= [token.text for token in doc if not token.is_stop]
    doc = nlp(' '.join(noStop))
    Gen1_noSTOP.append(doc)
        
for d in range(num_subs):
    text=(df['Gen_2'][d])
    doc=nlp(text)
    noStop= [token.text for token in doc if not token.is_stop]
    doc = nlp(' '.join(noStop))
    Gen2_noSTOP.append(doc)
    
for d in range(num_subs):
    text=(df['Gen_3'][d])
    doc=nlp(text)
    noStop= [token.text for token in doc if not token.is_stop]
    doc = nlp(' '.join(noStop))
    Gen3_noSTOP.append(doc)
    
for d in range(num_subs):
    text=(df['Gen_4'][d])
    doc=nlp(text)
    noStop= [token.text for token in doc if not token.is_stop]
    doc = nlp(' '.join(noStop))
    Gen4_noSTOP.append(doc)
    
for d in range(num_subs):
    text=(df['Gen_5'][d])
    doc=nlp(text)
    noStop= [token.text for token in doc if not token.is_stop]
    doc = nlp(' '.join(noStop))
    Gen5_noSTOP.append(doc)


#SPECIFY...
story=open("sara_noncanon.txt").read()
doc0=nlp(story)
noStop= [token.text for token in doc0 if not token.is_stop]
doc0_noStop = nlp(' '.join(noStop))


Gen0=[]
Gen0_noSTOP=[]
for s in range(num_subs):
    Gen0.append(doc0)
    Gen0_noSTOP.append(doc0_noStop)
    
AllStories_NoStop={'Gen0':Gen0_noSTOP,'Gen1':Gen1_noSTOP,'Gen2':Gen2_noSTOP,'Gen3':Gen3_noSTOP,'Gen4':Gen4_noSTOP, 'Gen5':Gen5_noSTOP}

##Rank Order Correlation Measures -- NONCANON
#MEASURE ONE: Across Generations

#Forward

AllSubs_Ordered_NOSTOP={}
AllSubs_Compare_NOSTOP={}



AllSubs_OrderedMatch_NOSTOP={}
AllSubs_CompareMatch_NOSTOP={} 

AllSubs_OrderedOriginal={}
AllSubs_CompareOriginal={}
AllSubs_HighestOriginal={}
AllSubs_LowestOriginal={}



doclist=[]

for x in range(num_subs): 
    SubName='Sub{}'.format(x+1)
    doclistNS=[AllStories_NoStop['Gen0'][x], AllStories_NoStop['Gen1'][x], AllStories_NoStop['Gen2'][x], AllStories_NoStop['Gen3'][x], AllStories_NoStop['Gen4'][x], AllStories_NoStop['Gen5'][x]]
    num_doc=len(doclistNS)
    
    sentences=[]
    num_sents=[]
    
    # Parse sentences for all docs
    for d in doclistNS:
        sentences.append(list(d.sents))
        num_sents.append(len(list(d.sents)))
    
    # Compare semantic similarities of sentences
    #Comparing Forward -- NO STOP WORDS
    Ordered_NOSTOP={}
    Compare_NOSTOP={} 
       
    for g in range(num_doc-1):
        similarity_sent=[[] for i in range(num_sents[g])]
        max_sim=[]
        for b in range(num_sents[g]): #for sentence in base doc
            docA=sentences[g][b]
            noStop= [token.text for token in docA if not token.is_stop]
            docA = nlp(' '.join(noStop))
            for c in range(num_sents[g+1]): #for sentence in compared doc
                docB=sentences[g+1][c] 
                noStop= [token.text for token in docB if not token.is_stop]
                docB = nlp(' '.join(noStop))
                similarity_sent[b].append(docA.similarity(docB))
            max_sim.append(1+similarity_sent[b].index(max(similarity_sent[b])))
        
        CompName='Gen{}vnext'.format(g)
        Ordered_NOSTOP[CompName]=max_sim #Find sentence most similar in each generation
        Compare_NOSTOP[CompName]=similarity_sent #Compare: Each gen, by each first gen sentence, by next gen sentence


    AllSubs_Ordered_NOSTOP[SubName]=Ordered_NOSTOP
    AllSubs_Compare_NOSTOP[SubName]=Ordered_NOSTOP 

    # Based on NO STOP Versions
OriGen=[[] for i in range(num_subs)]
FirstGen=[[] for i in range(num_subs)]
SecGen=[[] for i in range(num_subs)]
ThirGen=[[] for i in range(num_subs)]
FourGen=[[] for i in range(num_subs)]
OriCom=[[] for i in range(num_subs)]
FirstCom=[[] for i in range(num_subs)]
SecCom=[[] for i in range(num_subs)]
ThirCom=[[] for i in range(num_subs)]
FourCom=[[] for i in range(num_subs)]
OriCorr=[]
FirstCorr=[]
SecCorr=[]
ThirCorr=[]
FourCorr=[]
CorrTemp=[]

for s in range(num_subs):
    OriGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen0vnext']
    OriCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen0vnext']))))
    CorrTemp=spearmanr(OriCom[s],OriGen[s])
    OriCorr.append(CorrTemp[0])
    FirstGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vnext']
    FirstCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vnext']))))
    CorrTemp=spearmanr(FirstCom[s],FirstGen[s])
    FirstCorr.append(CorrTemp[0])
    SecGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vnext']
    SecCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vnext']))))
    CorrTemp=spearmanr(SecCom[s],SecGen[s])
    SecCorr.append(CorrTemp[0])
    ThirGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vnext']
    ThirCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vnext']))))
    CorrTemp=spearmanr(ThirCom[s],ThirGen[s])
    ThirCorr.append(CorrTemp[0])
    FourGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vnext']
    FourCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vnext']))))
    CorrTemp=spearmanr(FourCom[s],FourGen[s])
    FourCorr.append(CorrTemp[0])

    
Non_Forwards=OriCorr + FirstCorr + SecCorr +ThirCorr + FourCorr
Non_AllCorr=[OriCorr, FirstCorr, SecCorr, ThirCorr, FourCorr]
Non_df_AllCorr = pd.DataFrame(Non_AllCorr)
Non_df_AllCorr = pd.DataFrame.transpose(Non_df_AllCorr)
print(Non_df_AllCorr)
Non_df_AllCorr.to_excel("Exp4_NonCanon_OrderComparisonAcrossSubs_Spacy.xlsx",sheet_name='SentCorrelations')

Non_MeanCorr_Across=[statistics.mean(OriCorr), statistics.mean(FirstCorr),statistics.mean(SecCorr),statistics.mean(ThirCorr),statistics.mean(FourCorr)]
Non_OriCorr_Across=OriCorr
Non_FirstCorr_Across=FirstCorr
Non_SecCorr_Across=SecCorr
Non_ThirCorr_Across=ThirCorr
Non_FourCorr_Across=FourCorr

Non_StDError_Across=[stats.sem(OriCorr), stats.sem(FirstCorr),stats.sem(SecCorr),stats.sem(ThirCorr),stats.sem(FourCorr)]
Non_StDev_Across=[stats.tstd(OriCorr), stats.tstd(FirstCorr),stats.tstd(SecCorr),stats.tstd(ThirCorr),stats.tstd(FourCorr)]

objects = ('Original to Gen1', 'Gen1 to Gen2', 'Gen2 to Gen3', 'Gen3 to Gen4', 'Gen4 to Gen5')
y_pos = np.arange(len(objects))


plt.errorbar(y_pos*5, Non_MeanCorr_Across, yerr=Non_StDError_Across)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('NonCanon - Average Correlation Across Generations (Spacy)')
plt.savefig('Exp4_Non_Spacy_CorrOrder_Next')
plt.clf()


#Backward

## WHOLE NEW SECTION OF ANALYSES
AllSubs_Ordered_NOSTOP={}
AllSubs_Compare_NOSTOP={}



AllSubs_OrderedMatch_NOSTOP={}
AllSubs_CompareMatch_NOSTOP={} 

AllSubs_OrderedOriginal={}
AllSubs_CompareOriginal={}
AllSubs_HighestOriginal={}
AllSubs_LowestOriginal={}



doclist=[]

for x in range(num_subs): 
    SubName='Sub{}'.format(x+1)
    #doclist=[AllStories['Gen0'][x], AllStories['Gen1'][x], AllStories['Gen2'][x], AllStories['Gen3'][x], AllStories['Gen4'][x], AllStories['Gen5'][x]]
    doclistNS=[AllStories_NoStop['Gen0'][x], AllStories_NoStop['Gen1'][x], AllStories_NoStop['Gen2'][x], AllStories_NoStop['Gen3'][x], AllStories_NoStop['Gen4'][x], AllStories_NoStop['Gen5'][x]]
    num_doc=len(doclistNS)
    
    sentences=[]
    num_sents=[]
    
    # Parse sentences for all docs
    for d in doclistNS:
        sentences.append(list(d.sents))
        num_sents.append(len(list(d.sents)))
    
    # Compare semantic similarities of sentences
    Ordered_NOSTOP={}
    Compare_NOSTOP={} 
       
    for g in range(num_doc-1):
        similarity_sent=[[] for i in range(num_sents[g+1])]
        max_sim=[]
        for b in range(num_sents[g+1]): #for sentence in base doc
            docA=sentences[g+1][b]
            noStop= [token.text for token in docA if not token.is_stop]
            docA = nlp(' '.join(noStop))
            for c in range(num_sents[g]): #for sentence in compared doc
                docB=sentences[g][c] 
                noStop= [token.text for token in docB if not token.is_stop]
                docB = nlp(' '.join(noStop))
                similarity_sent[b].append(docA.similarity(docB))
            max_sim.append(1+similarity_sent[b].index(max(similarity_sent[b])))
        
        CompName='Gen{}vprevious'.format(g+1)
        Ordered_NOSTOP[CompName]=max_sim #Find sentence most similar in each generation
        Compare_NOSTOP[CompName]=similarity_sent #Compare: Each gen, by each first gen sentence, by next gen sentence


    AllSubs_Ordered_NOSTOP[SubName]=Ordered_NOSTOP
    AllSubs_Compare_NOSTOP[SubName]=Ordered_NOSTOP 

    # Based on NO STOP Versions
OriGen=[[] for i in range(num_subs)]
FirstGen=[[] for i in range(num_subs)]
SecGen=[[] for i in range(num_subs)]
ThirGen=[[] for i in range(num_subs)]
FourGen=[[] for i in range(num_subs)]
OriCom=[[] for i in range(num_subs)]
FirstCom=[[] for i in range(num_subs)]
SecCom=[[] for i in range(num_subs)]
ThirCom=[[] for i in range(num_subs)]
FourCom=[[] for i in range(num_subs)]
OriCorr=[]
FirstCorr=[]
SecCorr=[]
ThirCorr=[]
FourCorr=[]
CorrTemp=[]

for s in range(num_subs):
    OriGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vprevious']
    OriCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vprevious']))))
    CorrTemp=spearmanr(OriCom[s],OriGen[s])
    OriCorr.append(CorrTemp[0])
    FirstGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vprevious']
    FirstCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vprevious']))))
    CorrTemp=spearmanr(FirstCom[s],FirstGen[s])
    FirstCorr.append(CorrTemp[0])
    SecGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vprevious']
    SecCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vprevious']))))
    CorrTemp=spearmanr(SecCom[s],SecGen[s])
    SecCorr.append(CorrTemp[0])
    ThirGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vprevious']
    ThirCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vprevious']))))
    CorrTemp=spearmanr(ThirCom[s],ThirGen[s])
    ThirCorr.append(CorrTemp[0])
    FourGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5vprevious']
    FourCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5vprevious']))))
    CorrTemp=spearmanr(FourCom[s],FourGen[s])
    FourCorr.append(CorrTemp[0])

    
Non_Backwards=OriCorr + FirstCorr + SecCorr +ThirCorr + FourCorr
Non_AllCorr=[OriCorr, FirstCorr, SecCorr, ThirCorr, FourCorr]
Non_df_AllCorr = pd.DataFrame(Non_AllCorr)
Non_df_AllCorr = pd.DataFrame.transpose(Non_df_AllCorr)
print(Non_df_AllCorr)
Non_df_AllCorr.to_excel("Exp4_NonCanon_OrderComparisonAcrossSubs_Spacy_Backward.xlsx",sheet_name='SentCorrelations')

Non_MeanCorr_Across_Back=[statistics.mean(OriCorr), statistics.mean(FirstCorr),statistics.mean(SecCorr),statistics.mean(ThirCorr),statistics.mean(FourCorr)]
Non_OriCorr_Across_Back=OriCorr
Non_FirstCorr_Across_Back=FirstCorr
Non_SecCorr_Across_Back=SecCorr
Non_ThirCorr_Across_Back=ThirCorr
Non_FourCorr_Across_Back=FourCorr

Non_StDError_Across_Back=[stats.sem(OriCorr), stats.sem(FirstCorr),stats.sem(SecCorr),stats.sem(ThirCorr),stats.sem(FourCorr)]
Non_StDev_Across_Back=[stats.tstd(OriCorr), stats.tstd(FirstCorr),stats.tstd(SecCorr),stats.tstd(ThirCorr),stats.tstd(FourCorr)]

objects = ('Gen1 to Original', 'Gen2 to Gen1', 'Gen3 to Gen2', 'Gen4 to Gen3', 'Gen5 to Gen4')
y_pos = np.arange(len(objects))


plt.errorbar(y_pos*5, Non_MeanCorr_Across_Back, yerr=Non_StDError_Across_Back)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('NonCanon - Average Correlation Across Generations (Spacy) - Backward')
plt.savefig('Exp4_Non_Spacy_CorrOrder_Next_Backward')
plt.clf()



##GET CORRELATION OF TWO MEASURES

Non_Correlation_ForBack=pearsonr(Non_Forwards, Non_Backwards)
print(Non_Correlation_ForBack)

#########

#MEASURE TWO: TO LAST

#With Earlier Gens as Reference

##NONCANON - Corr Compared to Last
AllSubs_Ordered_NOSTOP={}
AllSubs_Compare_NOSTOP={}



AllSubs_OrderedMatch_NOSTOP={}
AllSubs_CompareMatch_NOSTOP={} 

AllSubs_OrderedOriginal={}
AllSubs_CompareOriginal={}
AllSubs_HighestOriginal={}
AllSubs_LowestOriginal={}



doclistNS=[]

for x in range(num_subs): 
    SubName='Sub{}'.format(x+1)
    #doclist=[AllStories['Gen0'][x], AllStories['Gen1'][x], AllStories['Gen2'][x], AllStories['Gen3'][x], AllStories['Gen4'][x], AllStories['Gen5'][x]]
    doclistNS=[AllStories_NoStop['Gen0'][x], AllStories_NoStop['Gen1'][x], AllStories_NoStop['Gen2'][x], AllStories_NoStop['Gen3'][x], AllStories_NoStop['Gen4'][x], AllStories_NoStop['Gen5'][x]]
    num_doc=len(doclistNS)
    
    sentences=[]
    num_sents=[]
    
    # Parse sentences for all docs
    for d in doclistNS:
        sentences.append(list(d.sents))
        num_sents.append(len(list(d.sents)))
    
    # Compare semantic similarities of sentences
    Ordered_NOSTOP={}
    Compare_NOSTOP={} 
       
    for g in range(num_doc-1):
        similarity_sent=[[] for i in range(num_sents[g])]
        max_sim=[]
        for b in range(num_sents[g]): #for sentence in base doc
            docA=sentences[g][b]
            noStop= [token.text for token in docA if not token.is_stop]
            docA = nlp(' '.join(noStop))
            for c in range(num_sents[5]): #for sentence in compared doc
                docB=sentences[5][c] 
                noStop= [token.text for token in docB if not token.is_stop]
                docB = nlp(' '.join(noStop))
                similarity_sent[b].append(docA.similarity(docB))
            max_sim.append(1+similarity_sent[b].index(max(similarity_sent[b])))
        
        CompName='Gen{}vnext'.format(g)
        Ordered_NOSTOP[CompName]=max_sim #Find sentence most similar in each generation
        Compare_NOSTOP[CompName]=similarity_sent #Compare: Each gen, by each first gen sentence, by next gen sentence


    AllSubs_Ordered_NOSTOP[SubName]=Ordered_NOSTOP
    AllSubs_Compare_NOSTOP[SubName]=Ordered_NOSTOP 

    # Based on NO STOP Versions
OriGen=[[] for i in range(num_subs)]
FirstGen=[[] for i in range(num_subs)]
SecGen=[[] for i in range(num_subs)]
ThirGen=[[] for i in range(num_subs)]
FourGen=[[] for i in range(num_subs)]
OriCom=[[] for i in range(num_subs)]
FirstCom=[[] for i in range(num_subs)]
SecCom=[[] for i in range(num_subs)]
ThirCom=[[] for i in range(num_subs)]
FourCom=[[] for i in range(num_subs)]
OriCorr=[]
FirstCorr=[]
SecCorr=[]
ThirCorr=[]
FourCorr=[]
CorrTemp=[]

for s in range(num_subs):
    OriGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen0vnext']
    OriCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen0vnext']))))
    CorrTemp=spearmanr(OriCom[s],OriGen[s])
    OriCorr.append(CorrTemp[0])
    FirstGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vnext']
    FirstCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vnext']))))
    CorrTemp=spearmanr(FirstCom[s],FirstGen[s])
    FirstCorr.append(CorrTemp[0])
    SecGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vnext']
    SecCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vnext']))))
    CorrTemp=spearmanr(SecCom[s],SecGen[s])
    SecCorr.append(CorrTemp[0])
    ThirGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vnext']
    ThirCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vnext']))))
    CorrTemp=spearmanr(ThirCom[s],ThirGen[s])
    ThirCorr.append(CorrTemp[0])
    FourGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vnext']
    FourCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vnext']))))
    CorrTemp=spearmanr(FourCom[s],FourGen[s])
    FourCorr.append(CorrTemp[0])
    

Non_MeanCorr_Last=[statistics.mean(OriCorr), statistics.mean(FirstCorr),statistics.mean(SecCorr),statistics.mean(ThirCorr),statistics.mean(FourCorr)]
Non_OriCorr_Last=OriCorr
Non_FirstCorr_Last=FirstCorr
Non_SecCorr_Last=SecCorr
Non_ThirCorr_Last=ThirCorr
Non_FourCorr_Last=FourCorr
Non_Error_Last=[stats.sem(OriCorr), stats.sem(FirstCorr),stats.sem(SecCorr),stats.sem(ThirCorr),stats.sem(FourCorr)]
Non_StDev_Last=[stats.tstd(OriCorr), stats.tstd(FirstCorr),stats.tstd(SecCorr),stats.tstd(ThirCorr),stats.tstd(FourCorr)]

objects = ('Original to Gen5', 'Gen1 to Gen5', 'Gen2 to Gen5', 'Gen3 to Gen5', 'Gen4 to Gen5')
y_pos = np.arange(len(objects))


plt.errorbar(y_pos*5, Non_MeanCorr_Last, yerr=Non_Error_Last)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('NonCanon- Average Correlation to Last Generation (Spacy)')
plt.savefig('Exp4_Non_Spacy_CorrOrdertoLast')
plt.clf()


Non_Last=Non_OriCorr_Last + Non_FirstCorr_Last + Non_SecCorr_Last + Non_ThirCorr_Last + Non_FourCorr_Last
Non_AllCorr_Last=[Non_OriCorr_Last, Non_FirstCorr_Last, Non_SecCorr_Last, Non_ThirCorr_Last, Non_FourCorr_Last]
Non_df_AllCorr_Last = pd.DataFrame(Non_AllCorr_Last)
Non_df_AllCorr_Last = pd.DataFrame.transpose(Non_df_AllCorr_Last)
print(Non_df_AllCorr_Last)
Non_df_AllCorr_Last.to_excel("Exp4_NonCanon_OrderComparisontoLast_Spacy.xlsx",sheet_name='SentCorrelations')

## With Last as Reference

AllSubs_Ordered_NOSTOP={}
AllSubs_Compare_NOSTOP={}



AllSubs_OrderedMatch_NOSTOP={}
AllSubs_CompareMatch_NOSTOP={} 

AllSubs_OrderedOriginal={}
AllSubs_CompareOriginal={}
AllSubs_HighestOriginal={}
AllSubs_LowestOriginal={}



doclistNS=[]

for x in range(num_subs): 
    SubName='Sub{}'.format(x+1)
    #doclist=[AllStories['Gen0'][x], AllStories['Gen1'][x], AllStories['Gen2'][x], AllStories['Gen3'][x], AllStories['Gen4'][x], AllStories['Gen5'][x]]
    doclistNS=[AllStories_NoStop['Gen0'][x], AllStories_NoStop['Gen1'][x], AllStories_NoStop['Gen2'][x], AllStories_NoStop['Gen3'][x], AllStories_NoStop['Gen4'][x], AllStories_NoStop['Gen5'][x]]
    num_doc=len(doclistNS)
    
    sentences=[]
    num_sents=[]
    
    # Parse sentences for all docs
    for d in doclistNS:
        sentences.append(list(d.sents))
        num_sents.append(len(list(d.sents)))
    
    # Compare semantic similarities of sentences
    #Comparing Forward -- NO STOP WORDS
    Ordered_NOSTOP={}
    Compare_NOSTOP={} 
       
    for g in range(num_doc-1):
        similarity_sent=[[] for i in range(num_sents[5])]
        max_sim=[]
        for b in range(num_sents[5]): #for sentence in base doc
            docA=sentences[5][b]
            noStop= [token.text for token in docA if not token.is_stop]
            docA = nlp(' '.join(noStop))
            for c in range(num_sents[g]): #for sentence in compared doc
                docB=sentences[g][c] 
                noStop= [token.text for token in docB if not token.is_stop]
                docB = nlp(' '.join(noStop))
                similarity_sent[b].append(docA.similarity(docB))
            max_sim.append(1+similarity_sent[b].index(max(similarity_sent[b])))
        
        CompName='Gen5v{}'.format(g)
        Ordered_NOSTOP[CompName]=max_sim #Find sentence most similar in each generation
        Compare_NOSTOP[CompName]=similarity_sent #Compare: Each gen, by each first gen sentence, by next gen sentence


    AllSubs_Ordered_NOSTOP[SubName]=Ordered_NOSTOP
    AllSubs_Compare_NOSTOP[SubName]=Ordered_NOSTOP 

    # Based on NO STOP Versions
OriGen=[[] for i in range(num_subs)]
FirstGen=[[] for i in range(num_subs)]
SecGen=[[] for i in range(num_subs)]
ThirGen=[[] for i in range(num_subs)]
FourGen=[[] for i in range(num_subs)]
OriCom=[[] for i in range(num_subs)]
FirstCom=[[] for i in range(num_subs)]
SecCom=[[] for i in range(num_subs)]
ThirCom=[[] for i in range(num_subs)]
FourCom=[[] for i in range(num_subs)]
OriCorr=[]
FirstCorr=[]
SecCorr=[]
ThirCorr=[]
FourCorr=[]
CorrTemp=[]

for s in range(num_subs):
    OriGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v0']
    OriCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v0']))))
    CorrTemp=spearmanr(OriCom[s],OriGen[s])
    OriCorr.append(CorrTemp[0])
    FirstGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v1']
    FirstCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v1']))))
    CorrTemp=spearmanr(FirstCom[s],FirstGen[s])
    FirstCorr.append(CorrTemp[0])
    SecGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v2']
    SecCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v2']))))
    CorrTemp=spearmanr(SecCom[s],SecGen[s])
    SecCorr.append(CorrTemp[0])
    ThirGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v3']
    ThirCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v3']))))
    CorrTemp=spearmanr(ThirCom[s],ThirGen[s])
    ThirCorr.append(CorrTemp[0])
    FourGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v4']
    FourCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5v4']))))
    CorrTemp=spearmanr(FourCom[s],FourGen[s])
    FourCorr.append(CorrTemp[0])
    

Non_MeanCorr_LastRef=[statistics.mean(OriCorr), statistics.mean(FirstCorr),statistics.mean(SecCorr),statistics.mean(ThirCorr),statistics.mean(FourCorr)]
Non_OriCorr_LastRef=OriCorr
Non_FirstCorr_LastRef=FirstCorr
Non_SecCorr_LastRef=SecCorr
Non_ThirCorr_LastRef=ThirCorr
Non_FourCorr_LastRef=FourCorr
Non_Error_LastRef=[stats.sem(OriCorr), stats.sem(FirstCorr),stats.sem(SecCorr),stats.sem(ThirCorr),stats.sem(FourCorr)]
Non_StDev_LastRef=[stats.tstd(OriCorr), stats.tstd(FirstCorr),stats.tstd(SecCorr),stats.tstd(ThirCorr),stats.tstd(FourCorr)]

objects = ('Original to Gen5', 'Gen1 to Gen5', 'Gen2 to Gen5', 'Gen3 to Gen5', 'Gen4 to Gen5')
y_pos = np.arange(len(objects))


plt.errorbar(y_pos*5, Non_MeanCorr_LastRef, yerr=Non_Error_LastRef)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('NonCanon- Average Correlation to Last Generation (Spacy) - LastRef')
plt.savefig('Exp4_Non_Spacy_CorrOrdertoLast_LastRef')
plt.clf()


Non_RefLast=Non_OriCorr_LastRef + Non_FirstCorr_LastRef + Non_SecCorr_LastRef + Non_ThirCorr_LastRef + Non_FourCorr_LastRef
Non_AllCorr_Last=[Non_OriCorr_LastRef, Non_FirstCorr_LastRef, Non_SecCorr_LastRef, Non_ThirCorr_LastRef, Non_FourCorr_LastRef]
Non_df_AllCorr_Last = pd.DataFrame(Non_AllCorr_Last)
Non_df_AllCorr_Last = pd.DataFrame.transpose(Non_df_AllCorr_Last)
print(Non_df_AllCorr_Last)
Non_df_AllCorr_Last.to_excel("Exp4_NonCanon_OrderComparisontoLast_Spacy_LastRef.xlsx",sheet_name='SentCorrelations')

##GET CORRELATION OF TWO MEASURES

Non_Correlation_RefLast=pearsonr(Non_Last, Non_RefLast)
print(Non_Correlation_RefLast)

#MEASURE THREE: TO INITIAL

##NONCANON - Corr Compared to Initial
AllSubs_Ordered_NOSTOP={}
AllSubs_Compare_NOSTOP={}



AllSubs_OrderedMatch_NOSTOP={}
AllSubs_CompareMatch_NOSTOP={} 

AllSubs_OrderedOriginal={}
AllSubs_CompareOriginal={}
AllSubs_HighestOriginal={}
AllSubs_LowestOriginal={}



doclistNS=[]

for x in range(num_subs): 
    SubName='Sub{}'.format(x+1)
    doclistNS=[AllStories_NoStop['Gen0'][x], AllStories_NoStop['Gen1'][x], AllStories_NoStop['Gen2'][x], AllStories_NoStop['Gen3'][x], AllStories_NoStop['Gen4'][x], AllStories_NoStop['Gen5'][x]]
    num_doc=len(doclistNS)
    
    sentences=[]
    num_sents=[]
    
    # Parse sentences for all docs
    for d in doclistNS:
        sentences.append(list(d.sents))
        num_sents.append(len(list(d.sents)))
    
    # Compare semantic similarities of sentences
    Ordered_NOSTOP={}
    Compare_NOSTOP={} 
       
    for g in range(1,num_doc):
        similarity_sent=[[] for i in range(num_sents[0])]
        max_sim=[]
        for b in range(num_sents[0]): #for sentence in base doc
            docA=sentences[0][b]
            noStop= [token.text for token in docA if not token.is_stop]
            docA = nlp(' '.join(noStop))
            for c in range(num_sents[g]): #for sentence in compared doc
                docB=sentences[g][c] 
                noStop= [token.text for token in docB if not token.is_stop]
                docB = nlp(' '.join(noStop))
                similarity_sent[b].append(docA.similarity(docB))
            max_sim.append(1+similarity_sent[b].index(max(similarity_sent[b])))
        
        CompName='FirstvGen{}'.format(g)
        Ordered_NOSTOP[CompName]=max_sim #Find sentence most similar in each generation
        Compare_NOSTOP[CompName]=similarity_sent #Compare: Each gen, by each first gen sentence, by next gen sentence


    AllSubs_Ordered_NOSTOP[SubName]=Ordered_NOSTOP
    AllSubs_Compare_NOSTOP[SubName]=Ordered_NOSTOP 

    # Based on NO STOP Versions
OriGen=[[] for i in range(num_subs)]
FirstGen=[[] for i in range(num_subs)]
SecGen=[[] for i in range(num_subs)]
ThirGen=[[] for i in range(num_subs)]
FourGen=[[] for i in range(num_subs)]
OriCom=[[] for i in range(num_subs)]
FirstCom=[[] for i in range(num_subs)]
SecCom=[[] for i in range(num_subs)]
ThirCom=[[] for i in range(num_subs)]
FourCom=[[] for i in range(num_subs)]
OriCorr=[]
FirstCorr=[]
SecCorr=[]
ThirCorr=[]
FourCorr=[]
CorrTemp=[]

for s in range(num_subs):
    OriGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen1']
    OriCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen1']))))
    CorrTemp=spearmanr(OriCom[s],OriGen[s])
    OriCorr.append(CorrTemp[0])
    FirstGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen2']
    FirstCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen2']))))
    CorrTemp=spearmanr(FirstCom[s],FirstGen[s])
    FirstCorr.append(CorrTemp[0])
    SecGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen3']
    SecCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen3']))))
    CorrTemp=spearmanr(SecCom[s],SecGen[s])
    SecCorr.append(CorrTemp[0])
    ThirGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen4']
    ThirCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen4']))))
    CorrTemp=spearmanr(ThirCom[s],ThirGen[s])
    ThirCorr.append(CorrTemp[0])
    FourGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen5']
    FourCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['FirstvGen5']))))
    CorrTemp=spearmanr(FourCom[s],FourGen[s])
    FourCorr.append(CorrTemp[0])
    

Non_MeanCorr_First=[statistics.mean(OriCorr), statistics.mean(FirstCorr),statistics.mean(SecCorr),statistics.mean(ThirCorr),statistics.mean(FourCorr)]
Non_OriCorr_First=OriCorr
Non_FirstCorr_First=FirstCorr
Non_SecCorr_First=SecCorr
Non_ThirCorr_First=ThirCorr
Non_FourCorr_First=FourCorr
Non_Error_First=[stats.sem(OriCorr), stats.sem(FirstCorr),stats.sem(SecCorr),stats.sem(ThirCorr),stats.sem(FourCorr)]
Non_StDev_First=[stats.tstd(OriCorr), stats.tstd(FirstCorr),stats.tstd(SecCorr),stats.tstd(ThirCorr),stats.tstd(FourCorr)]

objects = ('Original to Gen1', 'Original to Gen2', 'Original to Gen3', 'Original to Gen4', 'Original to Gen5')
y_pos = np.arange(len(objects))


plt.errorbar(y_pos*5, Non_MeanCorr_First, yerr=Non_Error_First)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('NonCanon- Average Correlation to Initial (Spacy)')
plt.savefig('Exp4_Non_Spacy_CorrOrdertoFirst_InitialReference')
plt.clf()


Non_First=Non_OriCorr_First + Non_FirstCorr_First+ Non_SecCorr_First+ Non_ThirCorr_First + Non_FourCorr_First
Non_AllCorr_First=[Non_OriCorr_First, Non_FirstCorr_First, Non_SecCorr_First, Non_ThirCorr_First, Non_FourCorr_First]
Non_df_AllCorr_First = pd.DataFrame(Non_AllCorr_First)
Non_df_AllCorr_First = pd.DataFrame.transpose(Non_df_AllCorr_First)
print(Non_df_AllCorr_First)
Non_df_AllCorr_First.to_excel("Exp4_NonCanon_OrderComparisontoFirst_Spacy.xlsx",sheet_name='SentCorrelations')


##NONCANON - Corr Compared to Initial, Other Reference

AllSubs_Ordered_NOSTOP={}
AllSubs_Compare_NOSTOP={}



AllSubs_OrderedMatch_NOSTOP={}
AllSubs_CompareMatch_NOSTOP={} 

AllSubs_OrderedOriginal={}
AllSubs_CompareOriginal={}
AllSubs_HighestOriginal={}
AllSubs_LowestOriginal={}



doclistNS=[]

for x in range(num_subs): 
    SubName='Sub{}'.format(x+1)
    #doclist=[AllStories['Gen0'][x], AllStories['Gen1'][x], AllStories['Gen2'][x], AllStories['Gen3'][x], AllStories['Gen4'][x], AllStories['Gen5'][x]]
    doclistNS=[AllStories_NoStop['Gen0'][x], AllStories_NoStop['Gen1'][x], AllStories_NoStop['Gen2'][x], AllStories_NoStop['Gen3'][x], AllStories_NoStop['Gen4'][x], AllStories_NoStop['Gen5'][x]]
    num_doc=len(doclistNS)
    
    sentences=[]
    num_sents=[]
    
    # Parse sentences for all docs
    for d in doclistNS:
        sentences.append(list(d.sents))
        num_sents.append(len(list(d.sents)))
    
    # Compare semantic similarities of sentences
    Ordered_NOSTOP={}
    Compare_NOSTOP={} 
       
    for g in range(1,num_doc):
        similarity_sent=[[] for i in range(num_sents[g])]
        max_sim=[]
        for b in range(num_sents[g]): #for sentence in base doc
            docA=sentences[g][b]
            noStop= [token.text for token in docA if not token.is_stop]
            docA = nlp(' '.join(noStop))
            for c in range(num_sents[0]): #for sentence in compared doc
                docB=sentences[0][c] 
                noStop= [token.text for token in docB if not token.is_stop]
                docB = nlp(' '.join(noStop))
                similarity_sent[b].append(docA.similarity(docB))
            max_sim.append(1+similarity_sent[b].index(max(similarity_sent[b])))
        
        CompName='Gen{}vFirst'.format(g)
        Ordered_NOSTOP[CompName]=max_sim #Find sentence most similar in each generation
        Compare_NOSTOP[CompName]=similarity_sent #Compare: Each gen, by each first gen sentence, by next gen sentence


    AllSubs_Ordered_NOSTOP[SubName]=Ordered_NOSTOP
    AllSubs_Compare_NOSTOP[SubName]=Ordered_NOSTOP 

    # Based on NO STOP Versions
OriGen=[[] for i in range(num_subs)]
FirstGen=[[] for i in range(num_subs)]
SecGen=[[] for i in range(num_subs)]
ThirGen=[[] for i in range(num_subs)]
FourGen=[[] for i in range(num_subs)]
OriCom=[[] for i in range(num_subs)]
FirstCom=[[] for i in range(num_subs)]
SecCom=[[] for i in range(num_subs)]
ThirCom=[[] for i in range(num_subs)]
FourCom=[[] for i in range(num_subs)]
OriCorr=[]
FirstCorr=[]
SecCorr=[]
ThirCorr=[]
FourCorr=[]
CorrTemp=[]

for s in range(num_subs):
    OriGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vFirst']
    OriCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen1vFirst']))))
    CorrTemp=spearmanr(OriCom[s],OriGen[s])
    OriCorr.append(CorrTemp[0])
    FirstGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vFirst']
    FirstCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen2vFirst']))))
    CorrTemp=spearmanr(FirstCom[s],FirstGen[s])
    FirstCorr.append(CorrTemp[0])
    SecGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vFirst']
    SecCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen3vFirst']))))
    CorrTemp=spearmanr(SecCom[s],SecGen[s])
    SecCorr.append(CorrTemp[0])
    ThirGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vFirst']
    ThirCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen4vFirst']))))
    CorrTemp=spearmanr(ThirCom[s],ThirGen[s])
    ThirCorr.append(CorrTemp[0])
    FourGen[s]=AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5vFirst']
    FourCom[s]=list(range(1,(1+len(AllSubs_Ordered_NOSTOP['Sub{}'.format(s+1)]['Gen5vFirst']))))
    CorrTemp=spearmanr(FourCom[s],FourGen[s])
    FourCorr.append(CorrTemp[0])
    

Non_MeanCorr_First_Back=[statistics.mean(OriCorr), statistics.mean(FirstCorr),statistics.mean(SecCorr),statistics.mean(ThirCorr),statistics.mean(FourCorr)]
Non_OriCorr_First_Back=OriCorr
Non_FirstCorr_First_Back=FirstCorr
Non_SecCorr_First_Back=SecCorr
Non_ThirCorr_First_Back=ThirCorr
Non_FourCorr_First_Back=FourCorr
Non_Error_First_Back=[stats.sem(OriCorr), stats.sem(FirstCorr),stats.sem(SecCorr),stats.sem(ThirCorr),stats.sem(FourCorr)]
Non_StDev_First_Back=[stats.tstd(OriCorr), stats.tstd(FirstCorr),stats.tstd(SecCorr),stats.tstd(ThirCorr),stats.tstd(FourCorr)]

objects = ('Gen1 to Original', 'Gen2 to Original', 'Gen3 to Original', 'Gen4 to Original', 'Gen5 to Original')
y_pos = np.arange(len(objects))



plt.errorbar(y_pos*5, Non_MeanCorr_First_Back, yerr=Non_Error_First_Back)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('NonCanon- Average Correlation to Initial (Spacy) - OtherRef')
plt.savefig('Exp4_Non_Spacy_CorrOrdertoFirst_OtherReference')
plt.clf()



Non_First_OtherRef=Non_OriCorr_First_Back + Non_FirstCorr_First_Back + Non_SecCorr_First_Back + Non_ThirCorr_First_Back + Non_FourCorr_First_Back
Non_AllCorr_First=[Non_OriCorr_First_Back, Non_FirstCorr_First_Back, Non_SecCorr_First_Back, Non_ThirCorr_First_Back, Non_FourCorr_First_Back]
Non_df_AllCorr_First = pd.DataFrame(Non_AllCorr_First)
Non_df_AllCorr_First = pd.DataFrame.transpose(Non_df_AllCorr_First)
print(Non_df_AllCorr_First)
Non_df_AllCorr_First.to_excel("Exp4_NonCanon_OrderComparisontoFirst_Spacy_OtherRef.xlsx",sheet_name='SentCorrelations')


##GET CORRELATION OF TWO MEASURES

Non_Correlation_RefFirst=pearsonr(Non_First, Non_First_OtherRef)
print(Non_Correlation_RefFirst)


######################

###PRINT ALL CORRELATIONS:
correlations = [[Canon_Correlation_ForBack, Non_Correlation_ForBack], [Canon_Correlation_RefLast,Non_Correlation_RefLast],[Canon_Correlation_RefFirst, Non_Correlation_RefFirst]]
df_cor = pd.DataFrame(correlations, columns = ['Correlation Canon','Correlation NonCanon'], index = ['Across Generations','To Last','To Initial'])
print(df_cor)


######################

# Both Canon and NonCanon Graphed Together

#MEASURE ONE: Across Generations
#Forward

objects = ('Original to Gen1', 'Gen1 to Gen2', 'Gen2 to Gen3', 'Gen3 to Gen4', 'Gen4 to Gen5')
y_pos = np.arange(len(objects))

plt.errorbar(y_pos*5, Can_MeanCorr_Across, yerr=Can_StDError_Across)
plt.errorbar(y_pos*5, Non_MeanCorr_Across, yerr=Non_StDError_Across)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Both - Average Correlation Across Generations (Spacy)')
plt.savefig('Exp4_Spacy_CorrOrdertoAcross')
plt.clf()


#Backwards
objects = ('Gen1 to Original', 'Gen2 to Gen1', 'Gen3 to Gen2', 'Gen4 to Gen3', 'Gen5 to Gen4')
y_pos = np.arange(len(objects))

plt.errorbar(y_pos*5, Can_MeanCorr_Across_Back, yerr=Can_StDError_Across_Back)
plt.errorbar(y_pos*5, Non_MeanCorr_Across_Back, yerr=Non_StDError_Across_Back)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Both - Average Correlation Across Generations (Spacy)-Back')
plt.savefig('Exp4_Spacy_CorrOrdertoAcross_Backwards')
plt.clf()


#MEASURE TWO: To Last
#Forward

objects = ('Original to Gen5', 'Gen1 to Gen5', 'Gen2 to Gen5', 'Gen3 to Gen5', 'Gen4 to Gen5')
y_pos = np.arange(len(objects))

plt.errorbar(y_pos*5, Can_MeanCorr_Last, yerr=Can_Error_Last)
plt.errorbar(y_pos*5, Non_MeanCorr_Last, yerr=Non_Error_Last)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Both - Average Correlation to Last Generation (Spacy)')
plt.savefig('Exp4_Spacy_CorrOrdertoLast')
plt.clf()

#Backward

objects = ('Gen5 to Original', 'Gen5 to Gen2', 'Gen5 to Gen2', 'Gen5 to Gen3', 'Gen5 to Gen4')
y_pos = np.arange(len(objects))

plt.errorbar(y_pos*5, Can_MeanCorr_LastRef, yerr=Can_Error_LastRef)
plt.errorbar(y_pos*5, Non_MeanCorr_LastRef, yerr=Non_Error_LastRef)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Both - Average Correlation to Last Generation (Spacy) - LastRef')
plt.savefig('Exp4_Spacy_CorrOrdertoLast_LastRef')
plt.clf()


#MEASURE THREE: To First

#Forwards
objects = ('Original to Gen1', 'Original to Gen2', 'Original to Gen3', 'Original to Gen4', 'Original to Gen5')
y_pos = np.arange(len(objects))

plt.errorbar(y_pos*5, Can_MeanCorr_First, yerr=Can_Error_First)
plt.errorbar(y_pos*5, Non_MeanCorr_First, yerr=Non_Error_First)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Both - Average Correlation to First Generation (Spacy)')
plt.savefig('Exp4_Spacy_CorrOrdertoFirst_Initial Reference')
plt.clf()


#Backwards
objects = ('Gen1 to Original', 'Gen2 to Original', 'Gen3 to Original', 'Gen4 to Original', 'Gen5 to Original')
y_pos = np.arange(len(objects))

plt.errorbar(y_pos*5, Can_MeanCorr_First_Back, yerr=Can_Error_First_Back)
plt.errorbar(y_pos*5, Non_MeanCorr_First_Back, yerr=Non_Error_First_Back)
plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Both - Average Correlation to First Generation (Spacy) - Backwards')
plt.savefig('Exp4_Spacy_CorrOrdertoFirst_Backwards')
plt.clf()


## Average Similarity across Both References (for both conditions, and all three measures)
Can_df_AllCorr_other=pd.read_excel("Exp4_Canon_OrderComparisontoFirst_Spacy_RefOther.xlsx",sheet_name='SentCorrelations')
Can_df_AllCorr_First=pd.read_excel("Exp4_Canon_OrderComparisontoFirst_Spacy.xlsx",sheet_name='SentCorrelations')
Can_df_AllCorr_RefLast=pd.read_excel("Exp4_Canon_OrderComparisontoLast_Spacy_RefLast.xlsx",sheet_name='SentCorrelations')
Can_df_AllCorr_Last=pd.read_excel("Exp4_Canon_OrderComparisontoLast_Spacy.xlsx",sheet_name='SentCorrelations')
Can_df_AllCorr_back=pd.read_excel("Exp4_Canon_OrderComparisonAcrossSubs_Spacy_Backwards.xlsx",sheet_name='SentCorrelations')
Can_df_AllCorr_front=pd.read_excel("Exp4_Canon_OrderComparisonAcrossSubs_Spacy.xlsx",sheet_name='SentCorrelations')

Can_df_First = (Can_df_AllCorr_other + Can_df_AllCorr_First)/2
Can_df_First.to_excel("Exp4_Canon_OrderComparisontoFirst_Average_Spacy.xlsx")
Can_df_Last = (Can_df_AllCorr_RefLast + Can_df_AllCorr_Last)/2
Can_df_Last.to_excel("Exp4_Canon_OrderComparisontoLast_Average_Spacy.xlsx")
Can_df_Across = (Can_df_AllCorr_back + Can_df_AllCorr_front)/2
Can_df_Across.to_excel("Exp4_Canon_OrderComparisonAcrossSubs_Average_Spacy.xlsx")

Non_df_AllCorr_other=pd.read_excel("Exp4_NonCanon_OrderComparisontoFirst_Spacy_OtherRef.xlsx",sheet_name='SentCorrelations')
Non_df_AllCorr_First=pd.read_excel("Exp4_NonCanon_OrderComparisontoFirst_Spacy.xlsx",sheet_name='SentCorrelations')
Non_df_AllCorr_RefLast=pd.read_excel("Exp4_NonCanon_OrderComparisontoLast_Spacy_LastRef.xlsx",sheet_name='SentCorrelations')
Non_df_AllCorr_Last=pd.read_excel("Exp4_NonCanon_OrderComparisontoLast_Spacy.xlsx",sheet_name='SentCorrelations')
Non_df_AllCorr_back=pd.read_excel("Exp4_NonCanon_OrderComparisonAcrossSubs_Spacy_Backward.xlsx",sheet_name='SentCorrelations')
Non_df_AllCorr_front=pd.read_excel("Exp4_NonCanon_OrderComparisonAcrossSubs_Spacy.xlsx",sheet_name='SentCorrelations')

Non_df_First = (Non_df_AllCorr_other + Non_df_AllCorr_First)/2
Non_df_Last = (Non_df_AllCorr_RefLast + Non_df_AllCorr_Last)/2
Non_df_Across = (Non_df_AllCorr_back + Non_df_AllCorr_front)/2
Non_df_First.to_excel("Exp4_NonCanon_OrderComparisontoFirst_Average_Spacy.xlsx")
Non_df_Last.to_excel("Exp4_NonCanon_OrderComparisontoLast_Average_Spacy.xlsx")
Non_df_Across.to_excel("Exp4_NonCanon_OrderComparisonAcrossSubs_Average_Spacy.xlsx")

#PLOT 

## Across

Can_MeanCorr_Across=[statistics.mean(Can_df_Across[0]), statistics.mean(Can_df_Across[1]),statistics.mean(Can_df_Across[2]),statistics.mean(Can_df_Across[3]),statistics.mean(Can_df_Across[4])]
print(Can_MeanCorr_Across)
Can_StDError_Across=[stats.sem(Can_df_Across[0]), stats.sem(Can_df_Across[1]),stats.sem(Can_df_Across[2]),stats.sem(Can_df_First[3]),stats.sem(Can_df_Across[4])]
Can_StD_Across=[np.std(Can_df_Across[0]), np.std(Can_df_Across[1]),np.std(Can_df_Across[2]),np.std(Can_df_First[3]),np.std(Can_df_Across[4])]
print(Can_StD_Across)

Non_MeanCorr_Across=[statistics.mean(Non_df_Across[0]), statistics.mean(Non_df_Across[1]),statistics.mean(Non_df_Across[2]),statistics.mean(Non_df_Across[3]),statistics.mean(Non_df_Across[4])]
print(Non_MeanCorr_Across)
Non_StDError_Across=[stats.sem(Non_df_Across[0]), stats.sem(Non_df_Across[1]),stats.sem(Non_df_Across[2]),stats.sem(Non_df_First[3]),stats.sem(Non_df_Across[4])]
Non_StD_Across=[np.std(Non_df_Across[0]), np.std(Non_df_Across[1]),np.std(Non_df_Across[2]),np.std(Non_df_First[3]),np.std(Non_df_Across[4])]
print(Non_StD_Across)

objects = ('Original to Gen1', 'Gen1 to Gen2', 'Gen2 to Gen3', 'Gen3 to Gen4', 'Gen4 to Gen5')
y_pos = np.arange(len(objects))


plt.errorbar(y_pos*5, Can_MeanCorr_Across, yerr=Can_StDError_Across, color='tab:blue')
plt.errorbar(y_pos*5, Non_MeanCorr_Across, yerr=Non_StDError_Across, color='tab:pink')
plt.ylim(0,1)
plt.xticks(y_pos*5)

plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Average Correlation Across Generations (Spacy)')
plt.savefig('Exp4_Spacy_CorrOrder_Across_Average')
plt.clf()


##To Initial

Can_MeanCorr_First=[statistics.mean(Can_df_First[0]), statistics.mean(Can_df_First[1]),statistics.mean(Can_df_First[2]),statistics.mean(Can_df_First[3]),statistics.mean(Can_df_First[4])]
Can_StDError_First=[stats.sem(Can_df_First[0]), stats.sem(Can_df_First[1]),stats.sem(Can_df_First[2]),stats.sem(Can_df_First[3]),stats.sem(Can_df_First[4])]
Non_MeanCorr_First=[statistics.mean(Non_df_First[0]), statistics.mean(Non_df_First[1]),statistics.mean(Non_df_First[2]),statistics.mean(Non_df_First[3]),statistics.mean(Non_df_First[4])]
Non_StDError_First=[stats.sem(Non_df_First[0]), stats.sem(Non_df_First[1]),stats.sem(Non_df_First[2]),stats.sem(Non_df_First[3]),stats.sem(Non_df_First[4])]

objects = ('Original to Gen1', 'Original to Gen2', 'Original to Gen3', 'Original to Gen4', 'Original to Gen5')
y_pos = np.arange(len(objects))


plt.errorbar(y_pos*5, Can_MeanCorr_First, yerr=Can_StDError_First)
plt.errorbar(y_pos*5, Non_MeanCorr_First, yerr=Non_StDError_First)

plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Average Correlation to Initial Story (Spacy)')
plt.savefig('Exp4_Spacy_CorrOrder_Initial_Average')
plt.clf()

##Last
Can_MeanCorr_Last=[statistics.mean(Can_df_Last[0]), statistics.mean(Can_df_Last[1]),statistics.mean(Can_df_Last[2]),statistics.mean(Can_df_Last[3]),statistics.mean(Can_df_Last[4])]
Can_StDError_Last=[stats.sem(Can_df_Last[0]), stats.sem(Can_df_Last[1]),stats.sem(Can_df_Last[2]),stats.sem(Can_df_Last[3]),stats.sem(Can_df_Last[4])]
Non_MeanCorr_Last=[statistics.mean(Non_df_Last[0]), statistics.mean(Non_df_Last[1]),statistics.mean(Non_df_Last[2]),statistics.mean(Non_df_Last[3]),statistics.mean(Non_df_Last[4])]
Non_StDError_Last=[stats.sem(Non_df_Last[0]), stats.sem(Non_df_Last[1]),stats.sem(Non_df_Last[2]),stats.sem(Non_df_Last[3]),stats.sem(Non_df_Last[4])]

objects = ('Original to Gen5', 'Gen1 to Gen5', 'Gen2 to Gen5', 'Gen3 to Gen5', 'Gen4 to Gen5')
y_pos = np.arange(len(objects))


plt.errorbar(y_pos*5, Can_MeanCorr_Last, yerr=Can_StDError_Last, color='tab:blue')
plt.errorbar(y_pos*5, Non_MeanCorr_Last, yerr=Non_StDError_Last, color='tab:pink')

plt.ylim(0,1)
plt.xticks(y_pos*5, objects)
plt.xlabel('Generation')
plt.ylabel('Order Correlation Measure')
plt.title('Average Correlation to Last Generation (Spacy)')
plt.savefig('Exp4_Spacy_CorrOrder_Last_Average')
plt.clf()