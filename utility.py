import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support
import time
from memory_profiler import memory_usage
import numpy as np
import torch

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
    Plots multiple AUC_ROC curves

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

def get_test_outputs(x_test, model):
  '''
  Gets the test outputs from the CNN model.

  :@param x_test: Test set images
  :@param model: CNN model
  :@return test_outputs torch vector
  '''
  model.eval() # set model to evaluation mode

  with torch.no_grad():
      test_outputs = model(x_test)

  return test_outputs

def get_metrics(x_test, y_test, model):
  '''
  Collect metrics: accuracy, runtime, peak memory usage, test outputs, y label predictions and f1 score

  :@param x_test: Test set images
  :@param x_test: Test set labels
  :@param model: CNN model
  :@return metrics
  '''
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  runtime, peak_mem, test_outputs = get_memory_usage_and_runtime(get_test_outputs, (x_test,model,))

  if isinstance(test_outputs, tuple):
      predictions, attention_weights = test_outputs
  else:
      predictions = test_outputs

  y_pred = torch.argmax(predictions, dim=1).tolist()
  accuracy = torch.sum(torch.tensor(y_pred).to(device) == y_test).item() / len(y_test)

  precision, recall, f1_score, support = precision_recall_fscore_support(y_test.cpu().numpy(), y_pred)
  average_f1_score = np.mean(f1_score)

  return accuracy, runtime, peak_mem, test_outputs, y_pred, average_f1_score
