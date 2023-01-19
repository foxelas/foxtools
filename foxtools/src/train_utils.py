# -*- coding: utf-8 -*

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pathlib


from contextlib import redirect_stdout
from keras.optimizers import Adam, RMSprop
from keras.losses import categorical_crossentropy, binary_crossentropy
from keras.metrics import Recall, Precision, FalseNegatives, FalsePositives, TrueNegatives, TruePositives
import segmentation_models as sm
from sklearn.metrics import roc_curve, auc
from keras.utils.vis_utils import plot_model
from os import mkdir
from os.path import join

if __name__ == "__main__":
    import fox_utils
else:
    from foxtools.src import fox_utils


############################### Save Settings ##############

def get_model_filename(suffix='', extension='txt', folder = None):
    model_name = ''

    # if folder is None:
    #     today = date.today()
    #     model_name = str(today) + '_'
    
    savedir = join(fox_utils.conf['Directories']['OutputDir'],
                   fox_utils.conf['Data Settings']['Dataset'], fox_utils.conf['Folder Names']['PythonTestFolderName'])
    if folder is not None:
        savedir = join(savedir, folder)

    try:
        mkdir(savedir)
    except OSError as error: 
        pass

    filename = join(savedir, model_name + suffix + '.' + extension)
    return filename

def save_model_summary(model, folder = None):
    filename = get_model_filename('modelsummary', 'txt', folder)
    if __name__ != "__main__":
        print("Saved at: ", filename)
    with open(filename, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            model.summary()

def save_model_graph(model, folder = None):
    filename = get_model_filename('modelgraph', 'png', folder)
    if __name__ != "__main__":
        print("Saved at: ", filename)
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)

def save_text(textstream, filename, folder = None):
    if (textstream != None): 
        filename = get_model_filename(filename, 'txt', folder)
        with open(filename, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                print(textstream)
    
def save_model_info(model, folder = None, optSettings = None):
    save_model_summary(model, folder)
    save_model_graph(model, folder)
    save_text(optSettings, 'optimizationSettings', folder)

    filename = get_model_filename('model', 'pkl', folder)
    abspath = pathlib.Path(filename).absolute()
    with open(str(abspath), 'wb') as f:
        pickle.dump(abspath, f)
        

########################################## COMPILE 
def get_compile_settings(learning_rate, optimizer, targetLoss, decay):
    
    lossFunName = "" 
    if type(targetLoss) == type(categorical_crossentropy):
        lossFunName = targetLoss.__name__
    elif str(targetLoss) == targetLoss:
        lossFunName = str(targetLoss)
    else:
        lossFunName = str(targetLoss._name)
        
    optSettings = "Compiled with" + "\n" + "Optimizer" + str(optimizer._name) + "\n" + "Learning Rate" + str(learning_rate) + "\n" + "Decay" + str(decay) + "\n" +  "Loss Function" + lossFunName
    return optSettings

def compile_and_save_structure(framework, model, optimizer, learning_rate, targetLoss, decay = 0):
     
    optSettings = get_compile_settings(learning_rate, optimizer, targetLoss, decay)
    metrics = [sm.metrics.iou_score, 'accuracy', Recall(), Precision(), FalseNegatives(), FalsePositives(), TrueNegatives(), TruePositives()]

    model.compile(
        optimizer = optimizer,  #'rmsprop', 'SGD', 'Adam',
        loss=targetLoss,
        metrics=metrics
        )

    folder = framework 

    save_model_info(model, folder, optSettings)

    return model

def compile_custom(framework, model, optimizerName = "Adam", learning_rate = 0.0001, decay=1e-06, lossFunction = "BCE+JC"):
    if optimizerName == "Adam":
        if decay == 0:
            optimizer = Adam(learning_rate=learning_rate, decay = decay)
        else:
            optimizer = Adam(learning_rate=learning_rate)
    elif optimizerName == "RMSProp":
        if decay == 0:
            optimizer = RMSprop(learning_rate=learning_rate, decay=decay) 
        else:
            optimizer = RMSprop(learning_rate=learning_rate)
    else: 
        optimizer = Adam(learning_rate=learning_rate)

    if lossFunction == "BCE+JC":
        target_loss = sm.losses.bce_jaccard_loss
    elif lossFunction == "BCE":
        target_loss = binary_crossentropy
    else: 
        target_loss = sm.losses.bce_jaccard_loss
        
    model = compile_and_save_structure(framework, model, optimizer, learning_rate, target_loss, decay)

    return model 

def fit_model(framework, model, x_train, y_train, x_test, y_test, numEpochs = 200, batchSize = 64):

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batchSize,
        epochs=numEpochs,
        validation_data=(x_test, y_test),
        )

    folder = framework
    plot_history(history, folder)

    return model, history

####################################### Evaluation ####################
def plot_history(history_info, folder = None):
    fig = plt.figure(1)
    plt.plot(history_info.history['loss'])
    plt.plot(history_info.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    filename = get_model_filename('loss', 'png', folder)
    plt.savefig(filename)

    plt.show()

    fig = plt.figure(2)
    plt.plot(history_info.history['iou_score'])
    plt.plot(history_info.history['val_iou_score'])
    plt.title('IOU scores')
    plt.ylabel('iou score')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    filename = get_model_filename('iou_score', 'png', folder)
    plt.savefig(filename)

    plt.show()

def visualize(hsi, gt, pred, folder = None, iou = None, suffix = None):
    fig = plt.figure(3)
    plt.clf()
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(fox_utils.get_display_image(hsi))

    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    plt.imshow(gt)

    plt.subplot(1,3,3)
    fig_title = "Prediction" if iou is None else "Prediction (" + str(iou) + "%)"
    plt.title(fig_title)
    plt.imshow(pred)

    filename = get_model_filename('v_' + suffix, 'png', folder)
    plt.savefig(filename)
    #plt.show()

    # plt.figure(4)
    # plt.clf()
    # plt.imshow(pred)
    # plt.axis('off')
    # plt.savefig(filename)

    filename = get_model_filename('p_' + suffix, 'mat', folder)
    savemat(filename,{"prediction": pred })


def plot_roc(fpr, tpr, auc_val, model_name, folder):

    fig = plt.figure(5)

    plt.clf()
    plt.plot([0, 1], [0, 1], 'k--')
    for (fpr_, tpr_, auc_val_, model_name_) in zip(fpr, tpr, auc_val, model_name):
        plt.plot(fpr_, tpr_, label=model_name_ + ' (area = {:.3f})'.format(auc_val_))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')

    # legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    filename = get_model_filename('auc', 'png', folder)
    plt.savefig(filename, bbox_inches = 'tight')
    plt.show()

    # Zoom in view of the upper left corner.
    fig = plt.figure(6)

    plt.clf()
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    for (fpr_, tpr_, auc_val_, model_name_) in zip(fpr, tpr, auc_val, model_name):
        plt.plot(fpr_, tpr_, label=model_name_ + ' (area = {:.3f})'.format(auc_val_))
    # plt.plot(fpr_rf, tpr_rf, label='RF (area = {:.3f})'.format(auc_rf))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    filename = get_model_filename('auc-zoom', 'png', folder)
    plt.savefig(filename, bbox_inches = 'tight')
    plt.show()


def calc_plot_roc(model, x_test, y_test, model_name, folder):
    y_scores = model.predict(x_test).ravel()
    y_test = np.reshape(y_test.astype(int), (y_scores.shape[0],  1))

    fpr, tpr, thresholds_keras = roc_curve(y_test, y_scores)
    auc_val = auc(fpr, tpr)

    plot_roc([fpr], [tpr], [auc_val], [model_name], folder)

    return fpr, tpr, auc_val

def get_eval_metrics(history, isValidation = False):
    target = ['accuracy', 'iou_score', 'precision', 'recall', 'false_positives', 'false_negatives', 'true_positives', 'true_negatives']
    if isValidation:
        target = [str('val_') + x for x in target]
    
    #print(history.history.keys())
    evalDict =  {x: history.history[x][-1] for x in target}
    return evalDict 

def get_eval_metrics_and_settings(history, isValidation, optz, lr, ed, lossFun):
    target = ['accuracy', 'iou_score', 'precision', 'recall', 'false_positives', 'false_negatives', 'true_positives', 'true_negatives']
    if isValidation:
        target = [str('val_') + x for x in target]
    
    #print(history.history.keys())
    eval_dict = {x: history.history[x][-1] for x in target}
    eval_dict["optimizer"] = optz
    eval_dict["learningRate"] = lr
    eval_dict["decay"] = ed
    eval_dict["lossFunction"] = lossFun

    return eval_dict

def evaluate_model(model, history, framework, folder, x_test, y_test):
    train_eval = get_eval_metrics(history)
    test_eval = get_eval_metrics(history, True)

    [fpr_, tpr_, auc_val_] = calc_plot_roc(model, x_test, y_test, framework, folder)

    return fpr_, tpr_, auc_val_, train_eval, test_eval

from scipy.io import savemat

def save_evaluate_model(model, history, framework, folder, x_test, y_test):
    fpr_, tpr_, auc_val_, train_eval, test_eval = evaluate_model(model, history, framework, folder, x_test, y_test)

    save_text(train_eval, 'results_train', folder)
    save_text(test_eval, 'results_test', folder)
    
    filename = get_model_filename('0_performance', 'mat', folder)
    
    mdic = {"fpr_": fpr_, "tpr_": tpr_, "auc_val_": auc_val_, "history": history.history, "train_eval": train_eval, "test_eval": test_eval}
    savemat(filename, mdic)

    print("Saved 0_performance.mat file.")
    return fpr_, tpr_, auc_val_


def save_evaluate_model_folds(folder, fpr_, tpr_, auc_val_, train_eval, test_eval, history):
    filename = get_model_filename('0_performance', 'mat', folder)
    mdic = {"fpr_": fpr_, "tpr_": tpr_, "auc_val_": auc_val_, "history": history, "train_eval": train_eval, "test_eval": test_eval}
    savemat(filename, mdic)

    return

def save_performance(folder, test_eval):
    filename = get_model_filename('0_performance', 'mat', folder)
    mdic = { "test_eval": test_eval}
    savemat(filename, mdic)

    return