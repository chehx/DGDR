import torch
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import logging

# validate the algorithm by AUC, accuracy and f1 score on val/test datasets

def algorithm_validate(algorithm, data_loader, writer, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []

        for image, label, domain, _ in data_loader:
            image = image.cuda()
            label = label.cuda().long()

            output = algorithm.predict(image)
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]

        acc = accuracy_score(label, pred)
        f1 = f1_score(label, pred, average='macro')
    
        auc_ovo = roc_auc_score(label, output, average='macro', multi_class='ovo')

        loss = loss / len(data_loader)

        if val_type in ['val', 'test']:
            writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
            writer.add_scalar('info/{}_loss'.format(val_type), loss, epoch)
            writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
            writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)     
                
            logging.info('{} - epoch: {}, loss: {}, acc: {}, auc: {}, F1: {}.'.format
            (val_type, epoch, loss, acc, auc_ovo, f1))

    algorithm.train()
    return auc_ovo, loss