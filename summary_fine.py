import matplotlib.pyplot as plt
import pandas as pd


def summ(file_loc, name, five):
    precision = []
    recall = []
    accuracy = []
    f1 = []
    labels = []
    checkpoints = [0,100,1847]
    if five:
        checkpoints = [0,1000, 2000, 3000, 3078]
    for i in checkpoints:
        file = file_loc+"/model-"str(i)+"/eval_results.txt"
        fo = open(file, "r")
        contents = fo.read()
        lines = contents.split("\n")
        precision.append(lines[1][4:])
        recall.append(lines[2][4:])
        f1.append(lines[3][4:]) 
        accuracy.append(lines[4][4:])
        labels.append(i)
    df = pd.DataFrame([precision, recall, f1, accuracy, labels] ,index=['precision', 'recall', 'f1', 'accuracy', 'labels'], columns = labels)# gca stands for 'get current axis'
    df=df.astype(float)
    df.to_csv(name+'.csv')
    print(df)
    # gca stands for 'get current axis'
    ax = plt.gca()
    df = df.T
    df.plot(kind='line',x='labels',y='accuracy',color='black',ax=ax)
    df.plot(kind='line',x='labels',y='precision', color='red', ax=ax)
    df.plot(kind='line',x='labels',y='recall', color='blue', ax=ax)
    df.plot(kind='line',x='labels',y='f1', color='green', ax=ax)

    plt.show()
    plt.savefig(name+'.png')
    
summ("./eval/x800", "x800_eval",False)
summ("./eval/800", "800_eval",False)
summ("./eval/0_5e", "0_5e_eval",True)