import config
from Net import Net
from joblib import dump, load
from train import extract_embeddings

import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc

import torch
from torch.utils.data import DataLoader, TensorDataset


def grid_search(embeddings, labels, data_id, loss_select):
    """
    Performs grid search to find the best SVM classifier for the given embeddings and labels.

    Parameters:
    - embeddings (np.array): The embeddings of the data.
    - labels (np.array): The labels for the data.
    - data_id (str): Identifier for the dataset.
    - loss_select (str): The loss function used for training.

    Returns:
    None
    """
    pca = PCA(n_components=0.9)
    scaler = StandardScaler()
    embeddings_reduced = pca.fit_transform(embeddings)
    embeddings_scaled = scaler.fit_transform(embeddings_reduced)

    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [1, 0.1, 0.01, 0.001],
    }

    svm = SVC()
    grid_search = GridSearchCV(svm, param_grid, verbose=3, scoring='recall_macro', n_jobs=-1)
    grid_search.fit(embeddings_scaled, labels)

    print(f'best_svm_model: {grid_search.best_estimator_}\n'
        f'best_score (recall macro): {grid_search.best_score_}\n'
        f'pca.n_components: {pca.n_components_}')

    dump(grid_search.best_estimator_, config.MODEL_OUT_DIR.joinpath(f'clf_{data_id}_{loss_select}_{config.EXP_NAME}.joblib'))
    dump(pca, config.MODEL_OUT_DIR.joinpath(f'pca_{data_id}_{loss_select}_{config.EXP_NAME}.joblib'))
    dump(scaler, config.MODEL_OUT_DIR.joinpath(f'scaler_{data_id}_{loss_select}_{config.EXP_NAME}.joblib'))


def plot_tsne(embeddings_2d, labels, title):
    """
    Plots a 2D t-SNE visualization of embeddings with labels.

    Parameters:
    - embeddings_2d (np.array): The 2D t-SNE embeddings.
    - labels (np.array): The labels corresponding to the embeddings.
    - title (str): Title for the plot.

    Returns:
    None
    """
    for label in set(labels):
        indices = labels == label
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=label)
    plt.legend()
    plt.title(title)
    plt.xlabel('1st Component')
    plt.ylabel('2nd Component')

def naive_eval(data_id, loss_select, x_test_loader):
    """
    Performs a naive evaluation of a model on a test set (classification metrics for entire set and t-SNE training).

    Parameters:
    - data_id (str): Identifier for the dataset.
    - loss_select (str): The loss function used for training.
    - x_test_loader (DataLoader): DataLoader for the test data.

    Returns:
    None
    """
    clf = load(config.MODEL_OUT_DIR.joinpath(f'clf_{data_id}_{loss_select}_{config.EXP_NAME}.joblib'))
    pca = load(config.MODEL_OUT_DIR.joinpath(f'pca_{data_id}_{loss_select}_{config.EXP_NAME}.joblib'))
    model = Net(config.EMBEDDING_SIZE, config.NET_IMG_SIZE).to(config.device)
    model.load_state_dict(torch.load(config.MODEL_OUT_DIR.joinpath(f'model_{data_id}_{loss_select}_{config.EXP_NAME}_{config.TIMESTAMP}.pth')))
    model.eval()

    test_embeddings, y_true_test = extract_embeddings(model, x_test_loader)

    scaler = StandardScaler()
    test_embeddings_scaled = scaler.fit_transform(pca.transform(test_embeddings))

    y_pred = clf.predict(test_embeddings_scaled)

    print(classification_report(y_true_test, y_pred, target_names=['Leish', 'No-Leish']))

    train_embeddings = np.load(config.MODEL_OUT_DIR.joinpath(f'train_embeddings_{data_id}_{loss_select}.npy'))
    train_embeddings_scaled = scaler.transform(pca.transform(train_embeddings))
    train_labels = np.load(config.MODEL_OUT_DIR.joinpath(f'train_labels_{data_id}_{loss_select}.npy'))

    val_embeddings = np.load(config.MODEL_OUT_DIR.joinpath(f'val_embeddings_{data_id}_{loss_select}.npy'))
    val_embeddings_scaled = scaler.transform(pca.transform(val_embeddings))
    val_labels = np.load(config.MODEL_OUT_DIR.joinpath(f'val_labels_{data_id}_{loss_select}.npy'))

    tsne = TSNE(n_components=2, random_state=config.SEED)

    train_embeddings_2d = tsne.fit_transform(train_embeddings)
    val_embeddings_2d = tsne.fit_transform(val_embeddings)
    test_embeddings_2d = tsne.fit_transform(test_embeddings)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plot_tsne(train_embeddings_2d, train_labels, 'Train')

    plt.subplot(1, 3, 2)
    plot_tsne(val_embeddings_2d, val_labels, 'Validation')

    plt.subplot(1, 3, 3)
    plot_tsne(test_embeddings_2d, y_pred, 'Test')

    plt.suptitle(f't-SNE Embedding Visualization [{loss_select}]')
    plt.show()

    # return test_embeddings_scaled, y_true_test, y_pred

def extract_data_as_np(data_loader):
    """
    Extracts all data and labels from a DataLoader and converts them into numpy arrays.

    Parameters:
    - data_loader (DataLoader): The DataLoader to extract data from.

    Returns:
    Tuple[np.array, np.array]: Arrays of data and labels.
    """
    all_data = []
    all_labels = []
    for inputs, labels in data_loader:
        all_data.extend(inputs.numpy())
        all_labels.extend(labels.numpy())

    return np.array(all_data), np.array(all_labels)

def get_avg_metrics(test_loader, model, scaler, clf, pca):
    """
    Calculates average classification metrics over multiple stratified folds of a test set.

    Parameters:
    - test_loader (DataLoader): DataLoader for the test data.
    - model (torch.nn.Module): Trained neural network model.
    - scaler (StandardScaler): Scaler for preprocessing embeddings.
    - clf (Classifier): Trained classifier.
    - pca (PCA): PCA for dimensionality reduction.

    Returns:
    None
    """
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True)

    metrics = {
        "precision_class_0": [],
        "precision_class_1": [],
        "recall_class_0": [],
        "recall_class_1": [],
        "f1_score_class_0": [],
        "f1_score_class_1": [],
        "accuracy": []
    }

    all_data, all_labels = extract_data_as_np(test_loader)

    for train_index, test_index in skf.split(all_data, all_labels):
        train_data, test_data = all_data[train_index], all_data[test_index]
        train_labels, test_labels = all_labels[train_index], all_labels[test_index]

        # fold_train_loader = DataLoader(TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_labels)), batch_size=config.BATCH_SIZE)
        fold_test_loader = DataLoader(TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_labels)), batch_size=config.BATCH_SIZE)

        fold_embeddings, fold_labels = extract_embeddings(model, fold_test_loader)
        fold_embeddings = scaler.transform(pca.transform(fold_embeddings))

        y_pred = clf.predict(fold_embeddings)
        report = classification_report(fold_labels, y_pred, output_dict=True, digits=4)

        metrics["precision_class_0"].append(report['0']['precision'])
        metrics["precision_class_1"].append(report['1']['precision'])
        metrics["recall_class_0"].append(report['0']['recall'])
        metrics["recall_class_1"].append(report['1']['recall'])
        metrics["f1_score_class_0"].append(report['0']['f1-score'])
        metrics["f1_score_class_1"].append(report['1']['f1-score'])
        metrics["accuracy"].append(report['accuracy'])


    metrics_avg = {metric: np.mean(values) for metric, values in metrics.items()}
    metrics_std = {metric: np.std(values) for metric, values in metrics.items()}

    for metric in metrics.keys():
        print(f"{metric} - Média: {metrics_avg[metric]:.4f}, Desvio Padrão: {metrics_std[metric]:.4f}")

############################################################################################################
## Advanced metrics

################################            ROC/AUC curve
def get_scores(x_test, model, pca, scaler, clf):
    """
    Calculates decision scores for the test data using a given model, PCA, scaler, and classifier.

    Parameters:
    - x_test (DataLoader): DataLoader for the test data.
    - model (torch.nn.Module): Trained neural network model.
    - pca (PCA): PCA used for dimensionality reduction of embeddings.
    - scaler (StandardScaler): Scaler for preprocessing embeddings.
    - clf (Classifier): Trained classifier.

    Returns:
    Tuple[np.array, np.array]: Test labels and decision scores.
    """
    test_embeddings, test_labels = extract_embeddings(model, x_test)
    test_embeddings = scaler.transform(pca.transform(test_embeddings))

    return test_labels, clf.decision_function(test_embeddings)

def plot_roc_curve(model_id, x_test, model, pca, scaler, clf):
    """
    Plots the individual ROC curve for a given set of true labels and decision scores.

    Parameters:
    - model_id (str): Identifier for the model used.
    - test_labels (np.array): True labels of the test data.
    - scores (np.array): Decision scores calculated for the test data.

    Returns:
    None
    """
    test_labels, scores = get_scores(x_test, model, pca, scaler, clf)
    fpr, tpr, thresholds = roc_curve(test_labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic [{model_id}]')
    plt.legend(loc="lower right")
    plt.show()

# models = [triplet_model, circle_model, multisim_model, npdot_model]
# pcas = [triplet_pca, circle_pca, multisim_pca, npdot_pca]
# scalers = [triplet_scaler, circle_scaler, multisim_scaler, npdot_scaler]
# clfs = [triplet_clf, circle_clf, multisim_clf, npdot_clf]

def plot_roc_curves(models, x_test, pcas, scalers, clfs):
    """
    Plots all ROC curves comparison for a given set of true labels and decision scores.

    Parameters:
    - models (str): .
    - x_test (DataLoader): DataLoader for the test data.
    - pcas (list): A list of fitted PCA objects used to reduce the dimensionality of
                   the test embeddings.
    - scalers (list): A list of fitted scaler objects used for normalizing the test
                      embeddings.
    - clfs (list): A list of classifier objects to calculate the scores needed for computing ROC curves.
    
    Returns:
    None
    """
    plt.figure()

    for (loss_id, model, pca, scaler, clf) in zip(config.LOSSES, models, pcas, scalers, clfs):
        test_labels, scores = get_scores(x_test, model, pca, scaler, clf)
        fpr, tpr, thresholds = roc_curve(test_labels, scores)
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, lw=2, label=f'Model {loss_id} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Comparison')
    plt.legend(loc="lower right")
    plt.show()
################################

################################            Matthew Correlation Coefficient (MCC)
def get_predictions(x_test, model, pca, scaler, clf):
    """
    Predicts labels for the test data using a given model, PCA, scaler, and classifier.

    Parameters:
    - x_test (DataLoader): DataLoader for the test data.
    - model (torch.nn.Module): Trained neural network model.
    - pca (PCA): PCA used for dimensionality reduction of embeddings.
    - scaler (StandardScaler): Scaler for preprocessing embeddings.
    - clf (Classifier): Trained classifier.

    Returns:
    Tuple[np.array, np.array]: Test labels and predicted labels.
    """
    test_embeddings, test_labels = extract_embeddings(model, x_test)
    test_embeddings = scaler.transform(pca.transform(test_embeddings))

    return test_labels, clf.predict(test_embeddings)
# y_true, y_pred_triplet = get_predictions(test_loader, triplet_model, triplet_pca, triplet_scaler, triplet_clf)
# mcc = matthews_corrcoef(y_true, y_pred_triplet)
################################

################################            Image Labelling and FP/FN counting
NUM_BATCHES = 4
NUM_IMGS = config.BATCH_SIZE*NUM_BATCHES

def get_labelled_img(model, scaler, pca, clf, data_loader):
    """
    Retrieves a batch of images and their labels from a DataLoader and predicts their labels using a given model, PCA, scaler, and classifier.

    Parameters:
    - model (torch.nn.Module): Trained neural network model.
    - scaler (StandardScaler): Scaler for preprocessing embeddings.
    - pca (PCA): PCA used for dimensionality reduction of embeddings.
    - clf (Classifier): Trained classifier.
    - data_loader (DataLoader): DataLoader to fetch data from.

    Returns:
    Tuple[Tensor, Tensor, np.array]: Batch of images, true labels, and predicted labels.
    """
    all_data, all_labels = [], []
    for i, (data, label) in enumerate(data_loader):
        if i >= NUM_BATCHES: break

        all_data.append(data)
        all_labels.append(label)

    data, labels = torch.cat(all_data), torch.cat(all_labels)
    dataloader_subset = DataLoader(TensorDataset(data, labels), batch_size=NUM_IMGS)
    embeddings, _ = extract_embeddings(model, dataloader_subset)
    embeddings = scaler.transform(pca.transform(embeddings))
    predictions = clf.predict(embeddings)

    return data, labels, predictions

def plot_images(loss_select, model, scaler, pca, clf, test_loader):
    """
    Plots a grid of images with their true and predicted labels, highlighting errors in red.

    Parameters:
    - images (Tensor): Batch of images.
    - loss_select (str): The loss function used for training.
    - true_labels (Tensor): True labels for each image.
    - predicted_labels (Tensor): Predicted labels for each image.

    Returns:
    Dict[str, int]: A dictionary containing the counts of total errors, false positives, and false negatives.
    """
    num_err = {'qtd_err':0, 'fp':0, 'fn':0}
    num_rows = NUM_IMGS // 4
    plt.figure(figsize=(10, 2*num_rows))

    images, true_labels, predicted_labels = get_labelled_img(model, scaler, pca, clf, test_loader)

    for i in range(len(images)):
        img = images[i].permute(1, 2, 0) # CxHxW to HxWxC

        plt.subplot(num_rows, 4, i + 1)
        plt.imshow(img)
        y_true = 'NO' if true_labels[i] else 'YES' # 0 - leish; 1 - no leish
        y_pred = 'NO' if predicted_labels[i] else 'YES'
        title_color = 'red' if y_true != y_pred else 'black'
        plt.title(f'true: {y_true}, pred: {y_pred}', color=title_color)
        plt.axis('off')

        if title_color == 'red':
            num_err['qtd_err'] += 1

            if true_labels[i]: # if 1 then model predicted 0
                num_err['fp'] += 1
            else: # if 0 then model predicted 1
                num_err['fn'] += 1

    plt.tight_layout()
    # plt.show()

    plt.savefig(config.PLOTS_MODEL_EVAL_OUT.joinpath(f'imgs_{config.DATA_ID}_{loss_select}.png'))
    plt.close()

    return num_err
# imgs, y_true, y_pred = get_labelled_img(triplet_model, triplet_scaler, triplet_pca, triplet_clf, test_loader)
# num_err = plot_images(imgs, y_true, y_pred)
################################




