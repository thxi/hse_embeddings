# have to use this, otherwise pytorch fails in jupyter due to
# CUDNN_STATUS_EXECUTION_FAILED
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from pytorch_metric_learning import losses, miners, distances, reducers, samplers
from sklearn.model_selection import train_test_split

from code.dataloader import AgeGroupMLDataset, AgeGroupClfDataset
from code.encoder_gru import Encoder
from code.classifier import Classifier
from code.utils import train_ml_model, train_classifier

sns.set()

sns.set_style("whitegrid", {'axes.grid': False})

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = 'cpu'

BATCH_SIZE = 16  # BATCH_SIZE unique persons
NUM_OF_SUBSEQUENCES = 5
SUBSEQUENCE_LENGTH = 90

EMBEDDING_DIM = 256
LR = 0.002
NUM_EPOCHS = 50

cat_vocab_sizes = [204]
cat_embedding_dim = 102
num_input_dim = 4
NUM_OBS = 30000

dataset = AgeGroupMLDataset()
dataset.load_client_to_indices()
clfdataset = AgeGroupClfDataset()
clfdataset.load_client_to_indices()

arch = 'GRU'

EMBEDDING_DIM = 256
nums_epochs = [100, 100, 50, 50, 20, 20, 20]
nums_obs = [300, 600, 1300, 2700, 5300, 10800, 21600]
nums_epochs = nums_epochs[::-1]
nums_obs = nums_obs[::-1]
accs = []

for NUM_OBS, NUM_EPOCHS in tqdm(zip(nums_obs, nums_epochs)):
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(NUM_OBS, NUM_EPOCHS)

    dataset.targets = dataset.targets[:NUM_OBS]
    clfdataset.targets = clfdataset.targets[:NUM_OBS]

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    train_idx, test_idx = train_test_split(np.arange(len(clfdataset.targets)),
                                           test_size=0.3,
                                           shuffle=True,
                                           stratify=clfdataset.targets,
                                           random_state=228)

    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

    trainloader = torch.utils.data.DataLoader(clfdataset,
                                              batch_size=BATCH_SIZE,
                                              sampler=train_sampler)
    testloader = torch.utils.data.DataLoader(clfdataset,
                                             batch_size=BATCH_SIZE,
                                             sampler=test_sampler)

    LR = 0.002

    # train decoder

    encoder = Encoder(
        numerical_input_dim=num_input_dim,
        cat_vocab_sizes=cat_vocab_sizes,
        cat_embedding_dim=cat_embedding_dim,
        embedding_dim=EMBEDDING_DIM,
    )
    encoder.to(device)
    encoder.train()
    optimizer = optim.Adam(encoder.parameters(), lr=LR)

    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)  # basically, returns average
    loss_func = losses.TripletMarginLoss(margin=0.4,
                                         distance=distance,
                                         reducer=reducer)
    mining_func = miners.TripletMarginMiner(margin=0.4,
                                            distance=distance,
                                            type_of_triplets="semihard")

    train_losses = train_ml_model(encoder, NUM_EPOCHS, dataloader,
                                  NUM_OF_SUBSEQUENCES, mining_func, loss_func,
                                  optimizer)
    fig, axs = plt.subplots(figsize=(12, 6))

    plt.plot(train_losses, label='train')
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title("final accuracy: {training}")
    plt.savefig(f'plots/ML_{arch}_{EMBEDDING_DIM}_{NUM_OBS}_{NUM_EPOCHS}.png')

    SCHEDULER_EPOCHS = 2
    LR = 0.002

    # train classifier decoder
    del optimizer, distance, reducer, loss_func, mining_func

    classifier = Classifier(numerical_input_dim=num_input_dim,
                            cat_vocab_sizes=cat_vocab_sizes,
                            cat_embedding_dim=cat_embedding_dim,
                            embedding_dim=EMBEDDING_DIM)
    classifier.encoder = encoder
    # classifier.freeze_encoder()
    classifier.to(device)

    optimizer = optim.Adam(classifier.decoder.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=SCHEDULER_EPOCHS,
    )

    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated(0))
    train_losses, train_accuracy, val_losses, val_accuracy = train_classifier(
        classifier,
        NUM_EPOCHS,
        trainloader,
        testloader,
        optimizer,
        criterion,
        scheduler,
        #         enable_train_mode=lambda: classifier.decoder.train(),
        #         enable_test_mode=lambda: classifier.decoder.eval(),
    )

    fig, axs = plt.subplots(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='validation')
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('iter')
    plt.ylabel('accuracy')
    plt.plot(train_accuracy, label='train')
    plt.plot(val_accuracy, label='validation')
    plt.legend()

    plt.savefig(
        f'plots/clfdec_{arch}_{EMBEDDING_DIM}_{NUM_OBS}_{NUM_EPOCHS}.png')

    accs.append(val_accuracy[-1])

plt.plot(nums_obs, accs)
plt.xlabel('iter')
plt.xscale('log', base=2)
plt.xticks(nums_obs)
plt.ylabel('accuracy')
plt.savefig(f'plots/clfdec_{arch}_numobs_to_acc_{NUM_OBS}_{NUM_EPOCHS}.png')