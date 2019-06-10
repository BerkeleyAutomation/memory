import matplotlib
matplotlib.use('Agg')
import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt

ground_dir = 'ground.txt'
pred_dir = 'pred.npy'

ground = open(ground_dir)
ground = ground.readlines()
ground = ground[0].split(' ')

ground_arr = [[], [], []]

for i in range(len(ground)):
    rem = i % 3
    ground_arr[rem] += [ground[i]]

pred_arr = np.load(pred_dir)
pred_arr = pred_arr - (sum(pred_arr) / float(len(pred_arr)))
pred_arr = pred_arr / np.var(pred_arr)
pred_arr = 1 / (1 + np.exp(-pred_arr))
pred_arr = pred_arr.tolist()

label = []
confidence = []
ground_arr[0].pop()

for i in range(len(pred_arr)):
    label.append(0 if int(ground_arr[0][i]) else 1)
    confidence.append(float(pred_arr[i][0]))
label = np.asarray(label)
confidence = np.asarray(confidence)
precision, recall, thresholds = sklearn.metrics.precision_recall_curve(label, confidence)
plt.figure()
plt.plot(recall, precision, 'b')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Siamese Precision Recall Curve')
plt.savefig('pr.png')

