from numpy import *
from datetime import datetime
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np

def evaluate(x, y, window=10000,):

    try:
        num = x.shape[0]
    except:
        num = len(x)

    # tp, fp, tn, fn, precision, recall, f1, gmean
    result = zeros((num, 8))

    result[window-1, 0] = sum([1 for i in range(window) if x[i] == 1 and y[i] == 1])
    result[window-1, 1] = sum([1 for i in range(window) if x[i] == 1 and y[i] == 0])
    result[window-1, 2] = sum([1 for i in range(window) if x[i] == 0 and y[i] == 0])
    result[window-1, 3] = sum([1 for i in range(window) if x[i] == 0 and y[i] == 1])

    if result[window-1, 0] + result[window-1, 1] == 0:
        result[window-1, 4] = 0
    else:
        result[window-1, 4] = result[window-1, 0]/(result[window-1, 0] + result[window-1, 1])

    if result[window-1, 0] + result[window-1, 3] == 0:
        result[window-1, 5] = 0
    else:
        result[window-1, 5] = result[window-1, 0]/(result[window-1, 0] + result[window-1, 3])

    if result[window-1, 4] == 0 and result[window-1, 5] == 0:
        result[window-1, 6] = 0
    else:
        result[window-1, 6] = 2 * result[window-1, 4] * result[window-1, 5]/(result[window-1, 4] + result[window-1, 5])

    if result[window-1, 5] == 0 or result[window-1, 2]+result[window-1, 1] == 0:
        result[window-1, 7] = 0
    else:
        result[window-1, 7] = sqrt(result[window-1, 5] * result[window-1, 2] / (result[window-1, 2]+result[window-1, 1]))

    for i in range(window, num):

        result[i, :] = result[i-1, :]
        if x[i] == 1:
            if y[i] == 1:
                result[i, 0] = result[i, 0] + 1
            else:
                result[i, 1] = result[i, 1] + 1
        else:
            if y[i] == 1:
                result[i, 3] = result[i, 3] + 1
            else:
                result[i, 2] = result[i, 2] + 1

        if x[i-window] == 1:
            if y[i-window] == 1:
                result[i, 0] = result[i, 0] - 1
            else:
                result[i, 1] = result[i, 1] - 1
        else:
            if y[i-window] == 1:
                result[i, 3] = result[i, 3] - 1
            else:
                result[i, 2] = result[i, 2] - 1

        if result[i, 0] + result[i, 1] == 0:
            result[i, 4] = 0
        else:
            result[i, 4] = result[i, 0]/(result[i, 0] + result[i, 1])

        if result[i, 0] + result[i, 3] == 0:
            result[i, 5] = 0
        else:
            result[i, 5] = result[i, 0]/(result[i, 0] + result[i, 3])

        if result[i, 4] == 0 and result[i, 5] == 0:
            result[i, 6] = 0
        else:
            result[i, 6] = 2 * result[i, 4] * result[i, 5]/(result[i, 4] + result[i, 5])

        if result[i, 5] == 0 or result[i, 2]+result[i, 1] == 0:
            result[i, 7] = 0
        else:
            result[i, 7] = sqrt(result[i, 5] * result[i, 2] / (result[i, 2]+result[i, 1]))

    """
    plt.plot(arange(num), result[:, 0], label="tp")
    plt.plot(arange(num), result[:, 1], label="fp")
    plt.plot(arange(num), result[:, 2], label="tn")
    plt.plot(arange(num), result[:, 3], label="fn")

    plt.legend()
    plt.show()
    plt.savefig('result//1dimension_metrics.png')
    plt.cla()
    plt.clf()
    plt.close()

    plt.plot(arange(num), result[:, 4], label="precision")
    plt.plot(arange(num), result[:, 5], label="recall")
    plt.plot(arange(num), result[:, 6], label="f1")
    plt.plot(arange(num), result[:, 7], label="g-mean")

    plt.legend()
    plt.show()
    plt.savefig('result//2dimension_metrics.png')
    plt.cla()
    plt.clf()
    plt.close()
    """

    return result

def Err1(x):

    e = sum(ones(x.shape)-x)
    return e

def Err2():

    return 1

def overall(x, y, attack, sampl, release_speed):

    try:
        num = x.shape[0]
    except:
        num = len(x)


    print(len(x))
    print(y.shape[0])
    temp1 = sum([1 for i in range(num) if x[i][1]==y[i]==1])
    print("True Positive"+str(temp1))
    temp2 = sum([1 for i in range(num) if x[i][1]==1 and y[i]==0])
    print("False Positive"+str(temp2))
    temp3 = sum([1 for i in range(num) if x[i][1]==y[i]==0])
    print("True Negative"+str(temp3))
    temp4 = sum([1 for i in range(num) if x[i][1]==0 and y[i]==1])
    print("False Negative"+str(temp4))
    try:
        TPR = temp1 / (temp1 + temp4)
    except ZeroDivisionError:
        TPR = 0
    try:
        TNR = temp3 / (temp3 + temp2)
    except ZeroDivisionError:
        TNR = 0
    try:
        FPR = temp2 / (temp2 + temp3)
    except ZeroDivisionError:
        FPR = 0
    try:
        FNR = temp4 / (temp4 + temp1)
    except ZeroDivisionError:
        FNR = 0
    try:
        temp5 = (temp1/(temp1+temp4)) if (temp1+temp4)!=0 else 0
    except ZeroDivisionError:
        temp5 = 0
    print("Recall: "+str(temp5))
    try:
        temp6 = (temp1/(temp1+temp2)) if (temp1+temp2)!=0 else 0
    except ZeroDivisionError:
        temp6 = 0
    print("Precision: "+str(temp6))
    try:
        temp7 = (temp1+temp3)/(temp1+temp2+temp3+temp4)
    except ZeroDivisionError:
        temp7 = 0
    print("Accuracy: "+str(temp7))
    try:
        temp8 = 2*temp5*temp6/(temp5+temp6)
    except ZeroDivisionError:
        temp8 = 0
    print("F1: "+str(temp8))

    score = []

    for a in (i[1] for i in x):
        score.append(a)

    print(y)
    print(y.shape[0])
    print(type(y))
    roc_curve_fpr, roc_curve_tpr, roc_curve_thres = metrics.roc_curve(y, score)
    roc_curve_fnr = 1 - roc_curve_tpr

    auc = metrics.roc_auc_score(y, score)
    eer = roc_curve_fpr[np.nanargmin(np.absolute((roc_curve_fnr - roc_curve_fpr)))]
    eer_sanity = roc_curve_fnr[np.nanargmin(np.absolute((roc_curve_fnr - roc_curve_fpr)))]

    print("AuC: "+str(auc))
    print("EER: "+str(eer))
    print("EER sanity: "+str(eer_sanity))

    # temp_9 = temp3/(temp3+temp2) if (temp3+temp2) != 0 else 0
    #print("Specity: "+str(temp_9))
    # temp9 = sqrt(temp_9*temp5)
    #print("G-mean: "+str(temp9))

    # save("result_overall_result.npy", [temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp_9, temp9])
        # Write the eval to a txt.
    ts_datetime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    f = open(f'{attack}-sampl-{sampl}-r-{release_speed}-{ts_datetime}-metrics.txt', 'a+')
    f.write(f'TP: {temp1}\n')
    f.write(f'TN: {temp3}\n')
    f.write(f'FP: {temp2}\n')
    f.write(f'FN: {temp4}\n')
    f.write(f'TPR: {TPR}\n')
    f.write(f'TNR: {TNR}\n')
    f.write(f'FPR: {FPR}\n')
    f.write(f'FNR: {FNR}\n')
    f.write(f'Accuracy: {temp7}\n')
    f.write(f'Precision: {temp6}\n')
    f.write(f'Recall: {temp5}\n')
    f.write(f'F1 Score: {temp8}\n')
    f.write(f'AuC: {auc}\n')
    f.write(f'EER: {eer}\n')
    f.write(f'EER sanity: {eer_sanity}\n')

