import matplotlib.pyplot as plt
import mplcyberpunk
import re

plt.style.use("cyberpunk")


def create_line_graph(nohup_file_path, number_of_linegraphs, graph_subjects,
                      patterns, nth_items, title, ylabel, xlabel, output_image_name):
    k = {}
    y_items = {}
    validation_epochs = []

    with open(nohup_file_path, 'r') as f:
        original_lines = f.readlines()

        for line in original_lines:
            if line.startswith('Training for stockdepth 0.25'):
                # set_1 = original_lines[0:original_lines.index(line)]
                set_2 = original_lines[original_lines.index(line):]
        sets = [set_2]
        # count = 0
        # for sample_set in sets:
        for linegraph in range(number_of_linegraphs):
            k[linegraph] = [line.strip('\n').replace('\t', ' ') for line in sets[0]
                            if line.strip('\n').startswith(graph_subjects[linegraph])]
            print(k[linegraph])
            y_items[linegraph] = [float(line.split(patterns[linegraph])[nth_items[linegraph]]) for line in k[linegraph]]
            y_average = sum(y_items[linegraph]) / len(y_items[linegraph])
            if graph_subjects[linegraph] == "Validation Loss":
                validation_epochs = [int(line.split(patterns[linegraph])[4]) for line in k[linegraph]]
        for i, y in y_items.items():
            if graph_subjects[i] == 'Validation Loss':
                x_items = validation_epochs
            else:
                x_items = [i for i in range(1, len(y) + 1)]
            print(y_average)
            plt.plot(x_items, y, linewidth=1.5, linestyle='--', label=f'{graph_subjects[i]}')
            plt.locator_params(integer=True)

    if number_of_linegraphs > 1:
        plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    mplcyberpunk.add_glow_effects()
    plt.savefig(output_image_name)
    plt.show()


def create_line_graph_for_kfold(nohup_file_path, number_of_linegraphs, graph_subjects, patterns, nth_items,
                                title, ylabel, xlabel, output_image_name):
    with open(nohup_file_path, 'r') as f:
        original_lines = f.readlines()
        fold_lines = list(filter(lambda a: 'Currently running' in a, original_lines))
        for fold_line in fold_lines:
            fold = int(re.findall(r'\d+', fold_line)[0])
            k = {}
            y_items = {}
            validation_epochs = []
            try:
                current_fold_lines = original_lines[original_lines.index(f'Currently running {str(fold)}\n'):
                                                    original_lines.index(f'Currently running {str(fold + 1)}\n')]
            except ValueError:
                current_fold_lines = original_lines[original_lines.index(f'Currently running {str(fold)}\n'):]

            for linegraph in range(number_of_linegraphs):
                k[linegraph] = [line.strip('\n').replace('\t', ' ') for line in current_fold_lines
                                if line.strip('\n').startswith(graph_subjects[linegraph])]

                y_items[linegraph] = [float(line.split(patterns[linegraph])[nth_items[linegraph]]) for line in
                                      k[linegraph]]
                if graph_subjects[linegraph] == "Validation Loss":
                    validation_epochs = [int(line.split(patterns[linegraph])[4]) for line in k[linegraph]]
            for i, y in y_items.items():
                if graph_subjects[i] == 'Validation Loss':
                    x_items = validation_epochs
                else:
                    x_items = range(1, len(y) + 1)
                plt.plot(x_items, y, linewidth=1.5, linestyle='--', label='Fold ' + str(fold) + ' ' + graph_subjects[i])
                plt.locator_params(integer=True)

    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    mplcyberpunk.add_glow_effects()
    plt.savefig(output_image_name)
    plt.show()


# for training vs validation loss
# create_line_graph('../aquarobotics/nohup_exp_13_2.out', 2, ['Training loss', 'Validation Loss'], [' ', ' '], [2, 6],
#                   'Training Vs Validation Loss in EXP 13 set 2', 'Loss', 'Epochs',
#                   'training_vs_val_loss_exp_13_2.jpg')

create_line_graph('nohup.out', 2, ['Training loss', 'Validation Loss'], [' ', ' '], [2, 6],
                  'Training Vs Validation in  experiment 24', 'Loss', 'Epochs', '24.jpg')
# for k fold
# create_line_graph_for_kfold('../aquarobotics/nohup_exp_11.out',  2, ['Training loss', 'Validation Loss'], [' ', ' '],
#                             [2, 6],  'Training Vs Validation Loss in EXP 11', 'Loss',
#                             'Epochs', 'training_vs_val_loss_exp_11.jpg')

# for training loss in 3 epochs of experiment 2
# create_line_graph('../aquarobotics/nohup_exp_13.out', 1, ['Epoch 1/6'], [' '], [6],
#                   'Training  in EXP 13', 'Loss', 'Steps', 'training_exp_13.jpg')
