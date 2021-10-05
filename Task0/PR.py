import numpy as np
import matplotlib.pyplot as plt


def TP(arr, threshold):
    return arr[(arr[:, 0] >= threshold) & (arr[:, 1] == 0)].shape[0]


def TN(arr, threshold):
    return arr[(arr[:, 0] <= threshold) & (arr[:, 1] == 1)].shape[0]


def FP(arr, threshold):
    return arr[(arr[:, 0] >= threshold) & (arr[:, 1] == 1)].shape[0]


def FN(arr, threshold):
    return arr[(arr[:, 0] <= threshold) & (arr[:, 1] == 0)].shape[0]


def recall(probe_predicts, threshold):
    return TP(probe_predicts, threshold) / (TP(probe_predicts, threshold) + FN(probe_predicts, threshold))


def precision(probe_predicts, threshold):
    return TP(probe_predicts, threshold) / (TP(probe_predicts, threshold) + FP(probe_predicts, threshold))


def TPR(probe_predicts, threshold):
    return recall(probe_predicts, threshold)


def FPR(probe_predicts, threshold):
    return FP(probe_predicts, threshold) / (FP(probe_predicts, threshold) + TN(probe_predicts, threshold))


def roc(probe_predictions, y_test):
    pp = np.copy(probe_predictions)
    pp[:, 1] = y_test

    sortedPrb = pp[pp[:, 0].argsort()][::-1]
    TPRs = [0]
    FPRs = [0]
    for i in sortedPrb:
        TPRs.append(TPR(sortedPrb, i[0]))
        FPRs.append(FPR(sortedPrb, i[0]))
    TPRs.append(1)
    FPRs.append(1)
    area = np.trapz(TPRs, FPRs)
    plt.text(0.5, 0.1, 'ROC-AUC = ' + str(area), fontsize=10, color='black')
    plt.plot(FPRs, TPRs, color='green')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])


def pr(probe_predictions, y_test):
    pp = np.copy(probe_predictions)
    pp[:, 1] = y_test

    sortedPrb = pp[pp[:, 0].argsort()][::-1]
    PRs = [1]
    REs = [0]
    for i in sortedPrb:
        PRs.append(precision(sortedPrb, i[0]))
        REs.append(recall(sortedPrb, i[0]))
    PRs.append(0)
    REs.append(1)
    area = np.trapz(PRs, REs)
    plt.text(0.5, 0.1, 'PR-AUC = ' + str(area), fontsize=10, color='black')
    plt.plot(REs, PRs, color='blue')
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
