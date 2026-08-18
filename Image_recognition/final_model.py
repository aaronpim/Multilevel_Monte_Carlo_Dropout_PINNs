import os
import time
import torch
import numpy as np
import pandas as pd

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

from src.trainmodel import CONFIG_to_folder_path, load_all, train_model, default_config
from src.MLMC_eval import get_estimator

def make_config(**overwrites):
    CONFIG = {
        "seed": 0,
        "epochs": 20,
        "lr": 1e-3,
        "batch_size": 128,
        "drop_p": 0.05,
        "num_conv_layers": 3,
        "width" : 64,
        "MLP_width": 128,
        "kernal_size": 3,
        "padding": 1
        }
    CONFIG.update(overwrites)
    return CONFIG

def varmain(CONFIG, dropout_evals = 100, aleatoric = True, epistemic = False):
    with torch.no_grad():
        model, _, test_loader = load_all(CONFIG, device = 'cpu')
        values = torch.cat([image for image, _ in test_loader])[:256]
        model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
        model.train()
        if aleatoric and not epistemic:
            model.eval()
            outputvar = model(values).var(dim = 0)
        elif not aleatoric and epistemic:
            default_input = values[0].unsqueeze(0)
            eval_vec = []
            for _ in range(dropout_evals):
                eval_vec.append( model(default_input) )
            eval_vec  = torch.stack(eval_vec).flatten(start_dim=0, end_dim=1)
            outputvar = eval_vec.var(dim = 0)
        elif aleatoric and epistemic:
            eval_vec = []
            for _ in range(dropout_evals):
                eval_vec.append( model(values) )
            eval_vec  = torch.stack(eval_vec).flatten(start_dim=0, end_dim=1)
            outputvar = eval_vec.var(dim = 0)
        return outputvar

def final_model_train(num_evals = 21):
    output_vec = []
    var_vec = []
    for seed in range(num_evals):
        CONFIG = make_config(seed = seed)
        final_loss, final_hold_loss = train_model(CONFIG)
        output_vec.append([seed, final_loss, final_hold_loss])

        alevar  = varmain(CONFIG, aleatoric = True, epistemic = False)
        epivar  = varmain(CONFIG, aleatoric = False, epistemic = True)
        fullvar = varmain(CONFIG, aleatoric = True, epistemic = True)

        for i in range(10):
            var_vec.append({
                "seed": seed,
                "element_idx": i,
                "aleatoric_variance": alevar[i],
                "epistemic_variance": epivar[i],
                "total_variance": fullvar[i]
            })

        print("")
        print(f"Completed seed {seed}")
        print("")
    pd.DataFrame(output_vec, columns=["seed", "train_loss", "hold_loss"]).to_csv('plots/final_models.csv', index=False)
    pd.DataFrame(var_vec).to_csv('plots/final_vars.csv', index=False)

def plot_NMIST_data():
    CONFIG = make_config()
    _, _, test_loader = load_all(CONFIG, device = 'cpu')
    all_labels = torch.cat([label for _, label in test_loader])
    all_images = torch.cat([image for image, _ in test_loader])
    counter = 0
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    for i, label in enumerate(all_labels):
        if int(label) == counter:
            matrix = all_images[i].squeeze()
            ax = axes[counter]
            ax.imshow(matrix, cmap='gray', origin='upper')
            ax.set_xticks([])
            ax.set_yticks([])
            counter += 1
        if counter > 9:
            break
    plt.tight_layout()
    plt.savefig('plots/NMIST.png')
    plt.close()

def plot_outputs(num_evals = 21, num_drop = 100,  filename  = 'plots/data.pt', plotname  = ['plots/accuracy.png','plots/epi_acc_var.png']):
    if not os.path.exists(filename):
        CONFIG = make_config()
        _, _, test_loader = load_all(CONFIG, device = 'cpu')
        all_labels = torch.cat([label for _, label in test_loader])
        all_images = torch.cat([image for image, _ in test_loader])

        results = torch.zeros((num_evals, num_drop, 10))
        with torch.no_grad():
            output_tensor = []
            for seed in range(num_evals):
                CONFIG = make_config(seed = seed)
                model, _, _ = load_all(CONFIG, device = 'cpu')
                model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
                model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
                model.train()

                for d in range(num_drop):
                    print(f'Starting dropout {d} with seed {seed}')
                    outputs = model(all_images)
                    preds = outputs.argmax(dim=-1)
                    for i in range(10):
                        mask = (all_labels == i)
                        if mask.sum() > 0:
                            correct = (preds[mask] == i).float().mean()
                            results[seed, d, i] = correct

            torch.save(results, filename)
    else:
        results = torch.load(filename, weights_only = True)
        results = results.reshape( results.shape[0]*results.shape[1], results.shape[2] )
        data = [results[:, i].numpy() for i in range(10)]
        plt.figure(figsize=(12, 6))
        bp = plt.boxplot(data, positions=range(len(data)), showfliers=True, patch_artist=True)
        for box in bp['boxes']:
            box.set(facecolor="orange", alpha=0.6)
        for median in bp['medians']:
            median.set(color='orange', linewidth=2)
        plt.xlabel("Digit")
        plt.ylabel("Model accuracy")
        plt.grid(axis='y', alpha=0.3)
        M = results.mean(dim = 0)
        plt.plot( range(len(M)), M, color = 'black')
        plt.scatter( range(len(M)), M, color = 'black')
        plt.tight_layout()
        plt.savefig(plotname[0])
        plt.close()

        results = torch.load(filename, weights_only = True)
        V = results.var(dim = 1)
        data = [V[:, i].numpy() for i in range(10)]
        plt.figure(figsize=(12, 6))
        bp = plt.boxplot(data, positions=range(len(data)), showfliers=True, patch_artist=True)
        for box in bp['boxes']:
            box.set(facecolor="orange", alpha=0.6)
        for median in bp['medians']:
            median.set(color='orange', linewidth=2)
        plt.xlabel("Digit")
        plt.ylabel("Epistemic variance")
        plt.grid(axis='y', alpha=0.3)
        M = V.mean(dim = 0)
        plt.plot( range(len(data)), M, color = 'black')
        plt.scatter( range(len(data)), M, color = 'black')
        plt.tight_layout()
        plt.savefig(plotname[1])
        plt.close()


def plot_matrix(num_evals = 21, num_drop = 100,  filename  = 'plots/data_matrix.pt'):
    if not os.path.exists(filename):
        CONFIG = make_config()
        _, _, test_loader = load_all(CONFIG, device = 'cpu')
        all_labels = torch.cat([label for _, label in test_loader])
        all_images = torch.cat([image for image, _ in test_loader])

        results = torch.zeros((num_evals, num_drop, 10, 10), dtype=torch.int32)
        with torch.no_grad():
            output_tensor = []
            for seed in range(num_evals):
                CONFIG = make_config(seed = seed)
                model, _, _ = load_all(CONFIG, device = 'cpu')
                model_path = os.path.join(CONFIG_to_folder_path(CONFIG), 'model.pt')
                model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only = True))
                model.train()
                for d in range(num_drop):
                    print(f'Starting dropout {d} with seed {seed}')
                    outputs = model(all_images)
                    preds = outputs.argmax(dim=-1)
                    for i, l in enumerate(all_labels):
                        results[seed, d, l, int(preds[i])] += 1
            torch.save(results, filename)
    else:
        plt.figure(figsize=(11, 9))
        results = torch.load(filename, weights_only = True)
        summation = results.sum(dim = [0,1])
        data = summation/summation.sum(dim = 1)
        plt.pcolor(data, edgecolors='k', linewidths=4, cmap='magma_r', norm = LogNorm(vmin= data[data>0].min() , vmax=data.max() ))
        plt.xlabel('digit')
        plt.ylabel('prediction')
        plt.xticks(np.arange(10) + 0.5, labels=np.arange(10))
        plt.yticks(np.arange(10) + 0.5, labels=np.arange(10))
        plt.colorbar()
        plt.savefig(filename[:-3]+'.pdf')
        plt.close()
        for i in range(10):
            data[i,i] = -1
            print(i, torch.argmax(data[i,]))
if __name__ == "__main__":
    # CONFIG = make_config(seed = 0)
    # outputvar = varmain(CONFIG, dropout_evals = 100, aleatoric = True, epistemic = False)
    # print(outputvar)
    # outputvar = varmain(CONFIG, dropout_evals = 100, aleatoric = False, epistemic = True)
    # print(outputvar)
    # outputvar = varmain(CONFIG, dropout_evals = 100, aleatoric = True, epistemic = True)
    # print(outputvar)
    # final_model_train()
    # plot_outputs()
    plot_matrix()
