from tqdm.auto import tqdm
import torch


def train_ml_model(encoder, NUM_EPOCHS, dataloader, num_of_subsequences,
                   mining_func, loss_func, optimizer):
    """train function for metric learning"""

    # might introduce bugs if model is not fully on cpu or gpu
    device = next(encoder.parameters()).device

    train_losses = []

    for epoch in tqdm(range(NUM_EPOCHS)):
        encoder.train()
        epoch_losses = []
        for batch_idx, (sequences, labels) in enumerate(dataloader):
            n, c = sequences[0], sequences[1]
            n = n.to(device)
            c = c.to(device)

            labels = torch.repeat_interleave(labels, num_of_subsequences)
            labels = labels.to(device)

            embeddings = encoder(n, c)
            indices_tuple = mining_func(embeddings, labels)
            loss = loss_func(embeddings, labels, indices_tuple)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        epoch_loss = torch.tensor(epoch_losses).mean()
        train_losses.append(epoch_loss)

        print("Epoch {} Loss = {}".format(epoch, epoch_loss))

    return train_losses


def train_classifier(classifier,
                     NUM_EPOCHS,
                     trainloader,
                     testloader,
                     optimizer,
                     criterion,
                     scheduler,
                     enable_train_mode=None,
                     enable_test_mode=None):
    """train function for a classifier"""
    # yapf: disable
    if enable_train_mode == None:
        enable_train_mode = lambda clf: clf.train()
    if enable_test_mode == None:
        enable_test_mode = lambda clf: clf.eval()
    # yapf: enable

    # might introduce bugs if model is not fully on cpu or gpu
    device = next(classifier.parameters()).device
    train_losses = []
    train_accuracy = []
    val_losses = []
    val_accuracy = []

    for epoch in tqdm(range(NUM_EPOCHS)):
        enable_train_mode()
        correct = 0
        total = 0
        epoch_losses = []
        for (sequences, labels) in trainloader:
            n, c = sequences[0], sequences[1]
            n = n.to(device)
            c = c.to(device)
            labels = labels.to(device)

            outputs = classifier(n, c)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_labels = torch.argmax(outputs, dim=1)
            correct += sum(labels == pred_labels)
            total += len(n)
            epoch_losses.append(loss.item())

        train_acc = correct / total
        epoch_loss = torch.tensor(epoch_losses).mean()
        train_losses.append(epoch_loss)
        train_accuracy.append(train_acc.item())

        enable_test_mode()
        correct = 0
        total = 0
        epoch_val_losses = []
        for sequences, labels in testloader:
            n, c = sequences[0], sequences[1]
            n = n.to(device)
            c = c.to(device)
            labels = labels.to(device)

            with torch.no_grad():
                outputs = classifier(n, c)
                loss = criterion(outputs, labels)

                pred_labels = torch.argmax(outputs, dim=1)
                correct += sum(labels == pred_labels)
                total += len(n)
                epoch_val_losses.append(loss.item())

        val_acc = correct / total
        epoch_val_loss = torch.tensor(epoch_val_losses).mean()
        val_losses.append(epoch_val_loss)
        val_accuracy.append(val_acc.item())

        scheduler.step(epoch_val_loss)

        print(f'Epoch {epoch}, train acc: {train_acc}, val acc: {val_acc}')
        print(f'train loss: {epoch_loss}; val loss: {epoch_val_loss}')
