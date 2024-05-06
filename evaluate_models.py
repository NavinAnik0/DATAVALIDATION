import os
import csv
import shutil
import numpy as np
import boto3
import torch
from PIL import Image
import cv2
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
import mplcyberpunk
import pandas as pd
import time
from utills import load_model, build_eval_transformation, predict, break_image_in_four_parts, load_config_from_yaml, \
    postprocess_output, load_label_csv, load_cv2_image_from_s3, load_labels_from_df
plt.style.use("cyberpunk")


def generate_detailed_result_csv(model_name, result_folder, results):
    fieldnames = list(results[0].keys())
    visualize_results = open(f'{result_folder}/{model_name}_visualize_results.csv', 'w', encoding='UTF8')
    visual_writer = csv.DictWriter(visualize_results, fieldnames)
    visual_writer.writeheader()
    visual_writer.writerows(results)


def generate_detailed_result_row(image_path, probabilities, label, i=None):
    if i is not None:
        image_path = image_path + f'_{i}'
    if not isinstance(probabilities, dict):
        return {'Name': image_path, 'Actual Label': label,
                'High Growth': str(round(probabilities[0][0].item() * 100,
                                         2)) + '%',
                'Low Growth': str(round(probabilities[0][1].item() * 100,
                                        2)) + '%',
                'Medium Growth': str(round(probabilities[0][2].item() * 100,
                                           2)) + '%'}
    else:
        return {'Name': image_path, 'Actual Label': label,
                'No Holes': round(probabilities['hole'][0][0].item() * 100, 2),
                'Holes': round(probabilities['hole'][0][1].item() * 100, 2),
                'High Growth': round(probabilities['growth'][0][0].item() * 100, 2),
                'Low Growth': round(probabilities['growth'][0][1].item() * 100, 2),
                'Medium Growth': round(probabilities['growth'][0][2].item() * 100, 2)}


def generate_overview_result_csv(config, classwise_accurates, accurates, total_images,
                                 classwise_accuracies, precisions, recalls, f1s, result_folder,
                                 strategy):
    field_name = ['experiment', 'model', 'accuracy', 'test_method', 'precision', 'recall', 'f1']
    for class_labels in classwise_accurates[list(classwise_accurates.keys())[0]].keys():
        field_name.append(f"{class_labels}_accuracy")
    if os.path.exists(f'{result_folder}/results.csv'):
        f = open(f'{result_folder}/results.csv', 'a', encoding='UTF8')
        writer = csv.DictWriter(f, field_name)
    else:
        f = open(f'{result_folder}/results.csv', 'w', encoding='UTF8')
        writer = csv.DictWriter(f, field_name)
        writer.writeheader()
    print(classwise_accuracies)
    result_rows = list()
    for key, value in accurates.items():
        temp = {'experiment': config['experiment'],
                'model': key,
                'accuracy': round((value / total_images) * 100, 2) if  len(classwise_accuracies[key].keys())==3 else
                round((sum(classwise_accuracies[key].values())/len(classwise_accuracies[key].keys()) *100), 2),
                'test_method': strategy,
                'precision': precisions[key],
                'recall': recalls[key],
                'f1': f1s[key]}
        for label_class, class_value in classwise_accuracies[key].items():
            temp[f'{label_class}_accuracy'] = round(classwise_accuracies[key][label_class] * 100, 2)
        result_rows.append(temp)
    writer.writerows(result_rows)


def generate_confusion_matrix(gt, pred, classes, checkpoint, result_folder, i=""):
    cm = confusion_matrix(gt, pred)

    commat = pd.DataFrame(np.mat(cm), index=classes,
                          columns=classes)
    plt.figure()
    sns.heatmap(commat, annot=True, cmap="Blues", fmt='d')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f'{checkpoint} Confusion Matrix')
    plt.savefig(f'{result_folder}/{checkpoint}_{i}_cm.jpg')
    plt.show()
    return cm


def generate_precision_recall_graph(cm, classes, checkpoint, result_folder, i=None):
    precision = np.diag(cm) / np.sum(cm, axis=0)
    recall = np.diag(cm) / np.sum(cm, axis=1)
    # Prepare precision recall graph for each category
    plt.figure()
    pre_rec = pd.DataFrame({'classes': classes,
                            'precision': precision,
                            'recall': recall})
    tidy = pre_rec.melt(id_vars='classes').rename(columns=str.title)
    ax = sns.barplot(x='Classes', y='Value', hue='Variable', data=tidy)
    for patch in ax.patches:
        ax.text(patch.get_x() + patch.get_width() / 2., 0.2 * patch.get_height(),
                round(patch.get_height()*100, 2),
                ha='center', va='bottom', fontsize=14, fontweight= 'bold',  rotation=90, color='white')
    total_precision = np.mean(precision)
    total_recall = np.mean(recall)
    total_f1 = (2*total_precision*total_recall)/(total_recall+total_precision)
    plt.ylabel('Precision/Recall')
    plt.xlabel('Classes')
    mplcyberpunk.add_glow_effects()
    plt.title(f'Precison {round(total_precision*100, 2)}%   Recall {round(total_recall*100, 2)}% of {checkpoint}')
    plt.savefig(f'{result_folder}/{checkpoint}_{i}_prec_rec.jpg')
    plt.show()
    return total_precision, total_recall, total_f1


def generate_barplot_for_one_metric(metric_dict, metric_name, experiment, result_folder):
    plt.figure(figsize=(12, 8))
    metric_df = pd.DataFrame.from_dict(metric_dict, orient='index').reset_index()
    metric_df.columns = ['checkpoints', 'value']
    ax = sns.barplot(x='checkpoints', y='value', data=metric_df)
    for patch in ax.patches:
        ax.text(patch.get_x() + patch.get_width() / 2., 0.2 * patch.get_height(),
                str(round(patch.get_height() * 100, 2))+"%",
                ha='center', va='bottom', fontsize=14, fontweight='bold', rotation=90, color='white')
    plt.ylabel(metric_name.capitalize())
    plt.xlabel('Epochs')
    ax.set_xticklabels(labels = [checkpoint.split('.')[0] for checkpoint in metric_df['checkpoints'].tolist()], rotation=90)
    # for item in ax.get_xticklabels():
    #     item.set_tex
    #     item.set_rotation(90)
    mplcyberpunk.add_glow_effects()
    plt.title(f'{metric_name.capitalize()}s of Experiment {experiment}')
    plt.savefig(f'{result_folder}/exp_{experiment}_{metric_name}_bar_chart.jpg')
    plt.show()


def generate_roc(gt, pred_probs, classes, result_folder, checkpoint, counter=None):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        gt_i = [1 if t == i else 0 for t in gt]
        pred_probs_i = [probs[i] for probs in pred_probs]
        fpr[i], tpr[i], _ = roc_curve(gt_i, pred_probs_i)
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic for multi-class data')
    plt.legend(loc="lower right")
    plt.grid(True)
    mplcyberpunk.add_glow_effects()
    plt.savefig(f'{result_folder}/{checkpoint}_{counter}_auc.jpg')
    plt.show()


def generate_predictions_on_unlabeled_data_nd_save_into_folders(config_file_path, checkpoint_path, strategy):
    # Load config file
    config = load_config_from_yaml(config_file_path)

    # Prepare paths
    result_folder = os.path.join(config['result_folder'], str(config['experiment']) + "best_model")
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Initialize transformations
    tranformation = build_eval_transformation(config)

    # Initialize counters
    with torch.no_grad():
        # Initialize counter for each checkpoint
        start_time = time.time()
        checkpoint_full_path = config['checkpoints_path'] + checkpoint_path
        # Load model
        model = load_model(checkpoint_full_path, config, False)
        print(f'Model Loading Time {time.time() - start_time} seconds')

        for image_path in os.listdir(config["test_set_path"]):
            if image_path.split('.')[-1] == "jpg":
                source_image_path = os.path.join(config["test_set_path"], image_path)
                print(source_image_path)
                cv2image = cv2.imread(source_image_path)
                if strategy == 'break':
                    # break image in four parts
                    images = break_image_in_four_parts(cv2image)
                    probability_list = []
                    for i, image in enumerate(images):
                        probabilities = predict(image, tranformation, model, config['device'])
                        probability_list.append(probabilities)

                    # Prepare prediction from average probabilities of broken images
                    average_probability = torch.mean(torch.stack(probability_list, 1), dim=1)
                    final_prediction = postprocess_output(average_probability,
                                                          config['classes'])
                else:
                    # Load the image with pil
                    image = Image.fromarray(cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB))
                    probabilities = predict(image, tranformation, model, config['device'])
                    final_prediction = postprocess_output(probabilities, config['classes'])
                cv2.putText(cv2image, str(final_prediction['hole'] + " " + final_prediction['growth']), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (255, 0, 0), 4, cv2.LINE_AA)
                cv2.imwrite(f"{result_folder}/{image_path}", cv2image)

                # final_folder = os.path.join(result_folder, final_prediction)
                # if not os.path.exists(final_folder):
                #     os.makedirs(final_folder)
                # shutil.copy(source_image_path, final_folder)


def evaluate_models(config_file_path, strategy='break', detailed_csv=True):
    # Load config file
    config = load_config_from_yaml(config_file_path)
    s3 = boto3.client('s3')
    # Prepare paths
    # Checkpoint Paths
    model_in_aws = not os.path.exists(config['checkpoints_path'])
    # If the checkpoints are stored in local storage
    if model_in_aws:

        checkpoints = s3.list_objects_v2(
            Bucket=config['bucket'],
            Prefix=config['checkpoints_path'])
        checkpoints = {checkpoint_object['Key'].split('/')[2]: checkpoint_object['Key'] for checkpoint_object in
                       checkpoints['Contents']}

    # If the checkpoints are stored in S3
    else:
        checkpoints = {checkpoint: os.path.join(config['checkpoints_path'], checkpoint) for checkpoint in
                       os.listdir(config['checkpoints_path'])}
    # Testset Path
    test_set_in_aws = not os.path.exists(config['test_set_path'])
    if test_set_in_aws:
        paginator = s3.get_paginator('list_objects_v2')
        testset_path_pages = paginator.paginate(
            Bucket=config['bucket'],
            Prefix=config['test_set_path']
        )
        if config['test_set_type'] == 'csv':
            test_set_csv = load_label_csv(os.path.join(config['test_set_path'], 'labels.csv'), config)
            images_path = [test_folder_path['Key'] for test_folder_page in testset_path_pages for test_folder_path in
                           test_folder_page['Contents'] if
                           test_folder_path['Key'].split('.')[-1] != "csv" and test_folder_path['Key'].split('/')[
                               -1] != ""]
            labels = {image_path: load_labels_from_df(test_set_csv, image_path.split('/')[-1]) for
                      image_path in images_path}
        else:
            images_path = [test_folder_path['Key'] for test_folder_page in testset_path_pages for test_folder_path in
                           test_folder_page['Contents'] if test_folder_path['Key'].split('/') > 2]

            labels = {image_path: image_path.split('/')[-2] for image_path in images_path}
    else:
        if config['test_set_type'] == 'csv':
            test_set_csv = load_label_csv(os.path.join(config['test_set_path'], 'labels.csv'), config)
            images_path = [os.path.join(config['test_set_path'], test_set_csv.loc[i, 'image_path'])
                           for i in range(len(test_set_csv))]
            labels = {image_path: load_labels_from_df(test_set_csv, image_path.split('/')[-1])
                      for image_path in images_path}
        else:
            images_path = [os.path.join(config['test_set_path'], image_path)
                           for category_path in os.listdir(config['test_set_path'])
                           for image_path in os.listdir(os.path.join(config['test_set_path'], category_path))]
            labels = {image_path: image_path.split('/')[-2] for image_path in images_path}

    result_folder = os.path.join(config['result_folder'], str(config['experiment']))
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # Initialize transformations
    tranformation = build_eval_transformation(config)
    # Initialize counters
    accurates = {}
    classwise_accurates = {}
    classwise_images = {}
    classwise_accuracies = {}
    for value in labels.values():
        if not isinstance(value, tuple):
            classwise_images[value] = classwise_images[value] + 1 if value in list(classwise_images.keys()) else 1
        else:
            for i in range(2):
                classwise_images[value[i]] = classwise_images[value[i]] + 1 \
                    if value[i] in list(classwise_images.keys()) else 1
    print(f" Classwise images {classwise_images}")
    total_images = len(images_path)
    # Start evaluation
    with torch.no_grad():
        precisions = {}
        recalls = {}
        f1s = {}
        for checkpoint, checkpoint_path in checkpoints.items():
            print(checkpoint)
            # Initialize counter for each checkpoint
            gt = []
            preds = []
            pred_probs = []
            accurates[checkpoint] = 0
            classwise_accurates[checkpoint] = {key: 0 for key in classwise_images.keys()}
            if detailed_csv:
                visualize_results_list = []
            start_time = time.time()
            # Load model
            model = load_model(checkpoint_path, config, model_in_aws)
            print(f'Model Loading Time {time.time() - start_time} seconds')
            for image_path in images_path:
                cv2image = load_cv2_image_from_s3(image_path, config) if test_set_in_aws else cv2.imread(image_path,
                                                                                                         cv2.IMREAD_COLOR)
                label = labels[image_path]
                if strategy == 'break':
                    # break image in four parts
                    images = break_image_in_four_parts(cv2image)
                    probability_list = []
                    for i, image in enumerate(images):
                        probabilities = predict(image, tranformation, model, config['device'])
                        probability_list.append(probabilities)
                        # Generate detailed probability report lines  for each broken image
                        visualize_results_list.append(
                            generate_detailed_result_row(image_path, probabilities, label, i))

                    # Prepare prediction from average probabilities of broken images
                    average_probability = torch.mean(torch.stack(probability_list, 1), dim=1)
                    final_prediction = postprocess_output(average_probability,
                                                          config['classes'])
                    # Prepare probability list for cm
                    pred_probs.append(average_probability.squeeze().tolist())

                else:
                    # Load the image with pil
                    image = Image.fromarray(cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB))
                    probabilities = predict(image, tranformation, model, config['device'])
                    # Prepare probability list for roc
                    pred_probs.append([probabilities['hole'].squeeze().tolist(), probabilities['growth'].squeeze().tolist()])if isinstance(probabilities, dict) else pred_probs.append(probabilities.squeeze().tolist())

                    # Generate detailed probability report lines  for each  image
                    visualize_results_list.append(
                        generate_detailed_result_row(image_path, probabilities, label, None))
                    final_prediction = postprocess_output(probabilities, config['classes'])

                # Prepare ground truths and pred labels as numeric values for each image to be used in cm
                gt.append([config['classes']['hole'].index(label[0]), config['classes']['growth'].index(label[1])]) if isinstance(label, tuple) else gt.append(config['classes'].index(label))
                preds.append([config['classes']['hole'].index(final_prediction['hole']), config['classes']['growth'].
                              index(final_prediction['growth'])]) if isinstance(final_prediction, dict) else preds.append(config['classes'].index(final_prediction))

                # Calculate accurates
                if not isinstance(final_prediction, dict):
                    if str(final_prediction) == str(label):
                        accurates[checkpoint] += 1
                        classwise_accurates[checkpoint][label] += 1
                else:
                    i = 0
                    for key, value in final_prediction.items():
                        if label[i] == final_prediction[key]:
                            classwise_accurates[checkpoint][label[i]] += 1
                            accurates[checkpoint] += 1
                            i += 1
            print(f'Checkpoint processing time {(time.time() - start_time) / 60}min')
            print(classwise_accurates)
            print(accurates)

            # Generate detailed csv
            if detailed_csv:
                generate_detailed_result_csv(checkpoint, result_folder, visualize_results_list)
            if type(gt[0]) == list:
                gt = np.array(gt)
                preds = np.array((preds))
                pred_probs = np.array(pred_probs)
                total_precision = 0
                total_recall = 0
                total_f1 = 0
                for i in range(gt.shape[-1]):
                    gt_i = [gt[j][i] for j in range(len(gt))]
                    pred_i = [preds[j][i] for j in range(len(preds))]
                    pred_probs_i = [pred_probs[j][i] for j in range(len(pred_probs))]
                    classes = config['classes']['hole'] if max(gt_i) == 1 else config['classes']['growth']
                    cm = generate_confusion_matrix(gt_i, pred_i, classes, checkpoint, result_folder, i)
                    precision, recall, f1 = generate_precision_recall_graph(cm, classes, checkpoint, result_folder, i)
                    total_precision += precision
                    total_recall += recall
                    total_f1 += f1
                    generate_roc(gt_i, pred_probs_i, classes, result_folder, checkpoint, i)

                precisions[checkpoint] = total_precision/2
                recalls[checkpoint] = total_recall/2
                f1s[checkpoint] = total_f1/2

            else:
                cm = generate_confusion_matrix(gt, preds, config['classes'], checkpoint, result_folder, '')
                precision, recall, f1 = generate_precision_recall_graph(cm, config['classes'], checkpoint, result_folder, '')
                precisions[checkpoint] = precision
                recalls[checkpoint] = recall
                f1s[checkpoint] = checkpoint
                generate_roc(gt, pred_probs, classes, result_folder, checkpoint, '')
    # Generate overall accuracies of each class
    print(accurates)
    for key, value in accurates.items():
        print(key)
        print(value)
        classwise_accuracies[key] = {}
        for label_class, class_value in classwise_accurates[key].items():
            classwise_accuracies[key][label_class] = float(classwise_accurates[key][label_class] / classwise_images[label_class])
    # Generate overall precision, recall and f1 graph
    generate_barplot_for_one_metric(precisions,metric_name='precision', experiment=config['experiment'],
                                    result_folder=result_folder)
    generate_barplot_for_one_metric(recalls, metric_name='recall', experiment=config['experiment'],
                                    result_folder=result_folder)
    generate_barplot_for_one_metric(f1s, metric_name='f1', experiment=config['experiment'],
                                    result_folder=result_folder)

    # Generate overview csv
    generate_overview_result_csv(config, classwise_accurates, accurates, total_images, classwise_accuracies, precisions,
                                 recalls, f1s, result_folder,
                                 strategy)


if __name__ == '__main__':
    # evaluate_models('config_files/evaluation.yaml', 'no break', True)
    generate_predictions_on_unlabeled_data_nd_save_into_folders('config_files/evaluation.yaml',
                                                                'epoch_17.pth', 'no_break')
