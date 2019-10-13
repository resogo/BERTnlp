import matplotlib.pyplot as plt
import pandas as pd

precision = []
recall = []
accuracy = []
f1 = []
labels = []
for i in range(0,820000,20000):
    file = "./fine_out_5e/checkpoint_"+str(i)+"/eval_results.txt"
    fo = open(file, "r")
    contents = fo.read()
    lines = contents.split("\n")
    precision.append(lines[1][2:])
    recall.append(lines[2][2:])
    f1.append(lines[3][2:])
    accuracy.append(lines[4][2:])
    labels.append(i)
df = pd.DataFrame([precision, recall, f1, accuracy, labels] ,index=['precision', 'recall', 'f1', 'accuracy', 'labels'], columns = labels)# gca stands for 'get current axis'
df.to_csv('./eval_5e.csv')
# gca stands for 'get current axis'
ax = plt.gca()
df = df.T
df.plot(kind='line',x='labels',y='accuracy',color='black',ax=ax)
df.plot(kind='line',x='labels',y='precision', color='red', ax=ax)
df.plot(kind='line',x='labels',y='recall', color='blue', ax=ax)
df.plot(kind='line',x='labels',y='f1', color='green', ax=ax)

plt.show()
plt.savefig('pretraining_checkpoints)_5e.png')