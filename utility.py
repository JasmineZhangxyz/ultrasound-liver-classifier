import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix
import time
from memory_profiler import memory_usage
from numpy import np

def get_memory_usage_and_runtime(function, arguments):
    '''
    Calculates the memory and runtime of a function, and calls the function

    :@param function, arguments - the function and its arguments
    :@return runtime (float) in seconds
    :@return peak_mem (float) in MB
    :@return result - which is the output of the function
    '''
    s = time.time()
    mem, result = memory_usage((function, arguments), retval=True)
    e = time.time()
    runtime = e-s
    peak_mem = max(mem)

    return runtime, peak_mem, result

def auc_roc_curves(y_data, title):
    '''
    Plots multiple AUC_ROC_curves

    :@param y_data (list) of y_test (numpy array), y_pred (numpy array), color (string), label (string)
    :@param title (string)
    '''
    calcs = []

    # get roc and auc
    for (y_test, y_pred, color, label) in y_data:
      fp_rate, tp_rate, thresholds = roc_curve(y_test, y_pred)
      roc_auc = auc(fp_rate, tp_rate)
      calcs.append((fp_rate, tp_rate, roc_auc, color, label))

    # plot
    plt.figure(figsize=(8, 8))
    for (fp_rate, tp_rate, roc_auc, color, label) in calcs:
      plt.plot(fp_rate, tp_rate, color=color, label=f'{label} AUC = {roc_auc:.6f}')

    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title(f'ROC-AUC Curve for {title}')
    plt.legend(loc="lower right")
    plt.show()

def create_confusion_matrix(y_test, y_pred, title=""):
    '''
    Creates confusion matrix

    :@param y_test (numpy) test set labels
    :@param y_pred (numpy) prediction labels labels
    :@param title (string) title of confusion matrix plot
    '''
    confusion_mat = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(xticks=np.arange(confusion_mat.shape[1]),
          yticks=np.arange(confusion_mat.shape[0]),
          xlabel='Predicted label', ylabel='True label',
          title=f'Confusion Matrix {title}')
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            ax.text(j, i, str(confusion_mat[i, j]),
                    ha='center', va='center', color='white')
    plt.show()
