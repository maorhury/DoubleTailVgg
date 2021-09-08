from torch import nn
from torchvision import models
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
import os
from torch import nn
from skimage import io
import sys
from torchvision import transforms
sys.path.insert(0, "/home/OM/projects/facial_feature_impact_comparison/modelling")
from local_model_store import LocalModelStore
import pandas as pd
import numpy as np
import tqdm


class DoubleTailVgg16(nn.Module):

    def __init__(self, n_classes_first_race, n_classes_second_race, split_layer):
        super(DoubleTailVgg16, self).__init__()
        self.vgg_head = models.vgg16()
        self.vgg_first_tail = models.vgg16(num_classes=n_classes_first_race)
        self.vgg_second_tail = models.vgg16(num_classes=n_classes_second_race)
        
        self.vgg_head = self.vgg_head.cuda()
        self.vgg_first_tail = self.vgg_first_tail.cuda()
        self.vgg_second_tail = self.vgg_second_tail.cuda()
        
        self.__build_arch(split_layer)
        
        print("head of Architecture:")
        print(self.vgg_head)
        
        print("First Tail of archirecture:")
        print(self.vgg_first_tail)
        
        print("Second Tail of archirecture:")
        print(self.vgg_second_tail)
        

    def forward(self, x, mode='train', race_mode = 'first'):
           
        if mode == 'train':
            head_pred = self.vgg_head(x)
            if race_mode == 'first':
                return self.vgg_first_tail(head_pred)
            elif race_mode == 'second':
                return self.vgg_second_tail(head_pred)
      
        elif mode == 'eval':
            head_pred = self.vgg_head(x)
            
            # forward through the first tail to get fc7 layer result
            tmp_first_tail_classifier_saver = self.vgg_first_tail.classifier
            self.vgg_first_tail.classifier = nn.Sequential(*[self.vgg_first_tail.classifier[i] for i in range(5)]) # temporarily delete the last layer from the network
            self.vgg_first_tail.cuda()
            with torch.no_grad():
                first_tail_result = self.vgg_first_tail(head_pred) # forward through the truncated network
            self.vgg_first_tail.classifier = tmp_first_tail_classifier_saver # restore the full network
            
            # forward through the second tail to get fc7 layer result
            tmp_second_tail_classifier_saver = self.vgg_second_tail.classifier
            self.vgg_second_tail.classifier = nn.Sequential(*[self.vgg_second_tail.classifier[i] for i in range(5)]) # temporarily delete the last layer from the network
            self.vgg_second_tail.cuda()
            with torch.no_grad():
                second_tail_result = self.vgg_second_tail(head_pred) # forward through the truncated network
            self.vgg_second_tail.classifier = tmp_second_tail_classifier_saver # restore the full network
            
            return first_tail_result, second_tail_result
                  
    ################################################################ HOWWWW is it possible that i dont use [0] when access features/classifier???????????????????????? make sure this is because idan function of loading ##########################################
    ################# This is NOT GONNA WORK WITH SPLIT IN THE CLASSIFIER BECAUSE 5 (ABOVE) DOESN'T FIT###############################
    
    def __build_arch(self, split_layer):
        if split_layer == "start":
            self.vgg_head.features = nn.Sequential()
            self.vgg_head.classifier = nn.Sequential()

        if split_layer == 'conv1':
            self.vgg_head = nn.Sequential(*[self.vgg_head.features[i] for i in range(0, 5)])
            self.vgg_first_tail.features = nn.Sequential(*[self.vgg_first_tail.features[i] for i in range(5, 31)])
            self.vgg_second_tail.features = nn.Sequential(*[self.vgg_second_tail.features[i] for i in range(5, 31)])

        elif split_layer == 'conv2':
            self.vgg_head = nn.Sequential(*[self.vgg_head.features[i] for i in range(0, 10)])
            self.vgg_first_tail.features = nn.Sequential(*[self.vgg_first_tail.features[i] for i in range(10, 31)])
            self.vgg_second_tail.features = nn.Sequential(*[self.vgg_second_tail.features[i] for i in range(10, 31)])

        elif split_layer == 'conv3':
            self.vgg_head = nn.Sequential(*[self.vgg_head.features[i] for i in range(0, 17)])
            self.vgg_first_tail.features = nn.Sequential(*[self.vgg_first_tail.features[i] for i in range(17, 31)])
            self.vgg_second_tail.features = nn.Sequential(*[self.vgg_second_tail.features[i] for i in range(17, 31)])

        elif split_layer == 'conv4':
            self.vgg_head = nn.Sequential(*[self.vgg_head.features[i] for i in range(0, 24)])
            self.vgg_first_tail.features = nn.Sequential(*[self.vgg_first_tail.features[i] for i in range(24, 31)])
            self.vgg_second_tail.features = nn.Sequential(*[self.vgg_second_tail.features[i] for i in range(24, 31)])

        elif split_layer == 'conv5':
            self.vgg_head = nn.Sequential(*[self.vgg_head.features[i] for i in range(0, 31)])
            self.vgg_first_tail.features = nn.Sequential()
            self.vgg_second_tail.features = nn.Sequential()

        elif split_layer == 'fc6':
            self.vgg_head.features = nn.Sequential(*[self.vgg_head.features[i] for i in range(0, 31)])
            self.vgg_head.classifier = nn.Sequential(*[self.vgg_head.classifier[i] for i in range(0, 3)])
            self.vgg_first_tail.features = nn.Sequential()
            self.vgg_first_tail.classifier = nn.Sequential(*[self.vgg_first_tail.classifier[i] for i in range(3, 7)])
            self.vgg_second_tail.features = nn.Sequential()
            self.vgg_second_tail.classifier = nn.Sequential(*[self.vgg_second_tail.classifier[i] for i in range(3, 7)])

        elif split_layer == 'fc7':
            self.vgg_head.features = nn.Sequential(*[self.vgg_head.features[i] for i in range(0, 31)])
            self.vgg_head.classifier = nn.Sequential(*[self.vgg_head.classifier[i] for i in range(0, 6)])
            self.vgg_first_tail.features = nn.Sequential()
            self.vgg_first_tail.classifier = nn.Sequential(*[self.vgg_first_tail.classifier[i] for i in range(6, 7)])
            self.vgg_second_tail.features = nn.Sequential()
            self.vgg_second_tail.classifier = nn.Sequential(*[self.vgg_second_tail.classifier[i] for i in range(6, 7)])


class MyDataset(Dataset):

    def __init__(self, dataset_path, images_and_labels_df, transform):
        self.dataset_path = dataset_path
        self.images_and_labels_df = images_and_labels_df
        self.transform = transform

    def __getitem__(self, idx):
        image = io.imread(os.path.join(self.dataset_path, self.images_and_labels_df.iloc[idx]['image']))
        label = self.images_and_labels_df.iloc[idx]['label']
        
        jpg_to_pil = transforms.ToPILImage()
        
        return self.transform(jpg_to_pil(image)), label

    def __len__(self):
        return len(self.images_and_labels_df)

class VerificationDataset(Dataset):
    """
    Dataset for verification test
    gets the first images of each pair in a list, and their corresponding images (the second images of each pair) in another list
    gets also the path to location of all images and tranform method to apply on each image
    """

    def __init__(self, dataset_path, first_images, second_images, transform):
        self.dataset_path = dataset_path
        self.first_images = first_images
        self.second_images = second_images
        self.transform = transform
        self.num_pairs = len(first_images)

    def __getitem__(self, idx):
        """
        returns two tensors of image pair after transform
        returns also the file names of the each image in the pair
        """

        first_image = io.imread(os.path.join(self.dataset_path, self.first_images[idx]))
        second_image = io.imread(os.path.join(self.dataset_path, self.second_images[idx]))

        jpg_to_pil = transforms.ToPILImage()

        return self.transform(jpg_to_pil(first_image)), self.transform(jpg_to_pil(second_image)), self.first_images[idx], self.second_images[idx]

    def __len__(self):
        return self.num_pairs

def train_epoch(first_dataloader, second_dataloader, model, loss_fn, head_optimizer, tail_optimizer):

    size = len(first_dataloader.dataset) + len(second_dataloader.dataset)
    first_iter = iter(first_dataloader)
    second_iter = iter(second_dataloader)
    
    for i in range(1000):
        
        # get the next batch of each dataloader
        first_x, first_y = next(first_iter)
        second_x, second_y = next(second_iter)
        
        # load data to gpu
        first_x = first_x.cuda(non_blocking=True)
        first_y = first_y.cuda(non_blocking=True)
        second_x = second_x.cuda(non_blocking=True)
        second_y = second_y.cuda(non_blocking=True)
        
        # Compute prediction for each batch (batch for race)
        first_pred = model(first_x, race_mode = 'first')
        second_pred = model(second_x, race_mode = 'second')
        
        first_loss = loss_fn(first_pred, first_y)
        second_loss = loss_fn(second_pred, second_y)
        
        loss = first_loss + second_loss
        
        # Backpropagation
        head_optimizer.zero_grad()
        tail_optimizer.zero_grad()
        loss.backward()
        head_optimizer.step()
        tail_optimizer.step()
        

        # print details to user
        if i % 100 == 0:
            loss, current = loss.item(), i * (len(first_x) + len(second_x))
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_epoch(dataloader, model, loss_fn, race_mode):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
        
            X = X.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            
            pred = model(X, race_mode=race_mode)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"{race_mode} Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return 100 * correct, test_loss

def verification_test(trained_model, dir_pairs_path, pairs_file_name, dataset_path, test_type):

    first_images = []
    second_images = []

    # access all image pairs in verification test file
    pairs_file = open(dir_pairs_path+pairs_file_name, 'r')
    lines = pairs_file.readlines()
    for line in lines:
        images = line.split(" ")
        first_images.append(images[0])
        second_images.append(images[1].rstrip())

    df = pd.DataFrame(columns = ['names', 'fc7_first_tail', 'fc7_second_tail', 'type'])

    transform = transforms.Compose([transforms.Resize([256, 256]), transforms.CenterCrop([224, 224]), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    verification_dataset = VerificationDataset(dataset_path, first_images, second_images, transform)
    verification_dataloader = DataLoader(verification_dataset, batch_size=8)
    
    cos = nn.CosineSimilarity()
    
    # iterate over the batches and feed them into the trained model
    for first_images, second_images, first_images_names, second_images_names in verification_dataloader:
    
        # fc7 scores for all images
        first_images = first_images.cuda()
        second_images = second_images.cuda()
        first_images_preds_first_tail,  first_images_preds_second_tail= trained_model.forward(first_images, mode='eval')
        second_images_preds_first_tail,  second_images_preds_second_tail= trained_model.forward(second_images, mode='eval')
        
        first_tail_distance_scores = 1 - cos(first_images_preds_first_tail, second_images_preds_first_tail) # cos similarity of fc7 first tail results for each pairs 
        second_tail_distance_scores = 1 - cos(first_images_preds_second_tail, second_images_preds_second_tail) # cos similarity of fc7 second tail results for each pairs
        
        # insert new results into df for future analysis
        first_tail_distance_scores = first_tail_distance_scores.detach().cpu().numpy() # convert tensor to numpy
        second_tail_distance_scores = second_tail_distance_scores.detach().cpu().numpy() # convert tensor to numpy
        tmp_df = pd.DataFrame(np.array([[(first_images_names[i], second_images_names[i]), first_tail_distance_scores[i], second_tail_distance_scores[i], test_type]  for i in range(len(first_images))], dtype=object), columns = ['names', 'fc7_first_tail', 'fc7_second_tail', 'type'])
        df = pd.concat([df, tmp_df])
 
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    df.to_csv(f"/home/ssd_storage/experiments/students/OM/DoubleTailVgg_Exp/{test_type}.csv")

def train_DoubleTailVgg16_model(images_path, model_path, eths, n_classes_first_race, n_classes_second_race, split_layer,
                                epochs, train_perc, dir_pairs_path, test_types_mapped_to_pairs_file_name_dict, dataset_path, seed):
    # Initialize model
    model = DoubleTailVgg16(n_classes_first_race, n_classes_second_race, split_layer)

    # Initialize the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Initialize optimizer
    head_optimizer = torch.optim.SGD(model.vgg_head.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    head_stepLR = torch.optim.lr_scheduler.StepLR(head_optimizer, step_size=15, gamma=0.1)
    tails_params = list(model.vgg_first_tail.parameters()) + list(model.vgg_second_tail.parameters())
    tail_optimizer = torch.optim.SGD(tails_params, lr=0.01, momentum=0.9, weight_decay=5e-4)
    tail_stepLR = torch.optim.lr_scheduler.StepLR(tail_optimizer, step_size=15, gamma=0.1)

    # initialize train and test dataloaders
    first_eth_images_and_labels_df_train = pd.DataFrame(columns=['image', 'label'])
    first_eth_images_and_labels_df_test = pd.DataFrame(columns=['image', 'label'])
    second_eth_images_and_labels_df_train = pd.DataFrame(columns=['image', 'label'])
    second_eth_images_and_labels_df_test = pd.DataFrame(columns=['image', 'label'])
    
    for j, eth in enumerate(eths):
        i = 0
        for identity in os.listdir(os.path.join(images_path, eth)):
            id_images = f'{eth}/{identity}/' + pd.Series(os.listdir(os.path.join(images_path, eth, identity)))
            id_images_train = id_images.sample(frac=train_perc, random_state=seed)
            id_images_test = id_images.drop(id_images_train.index)
            if j==0:
                first_eth_images_and_labels_df_train = first_eth_images_and_labels_df_train.append(pd.DataFrame(data={'image': id_images_train, 'label': i}))
                first_eth_images_and_labels_df_test = first_eth_images_and_labels_df_test.append(pd.DataFrame(data={'image': id_images_test, 'label': i}))
            elif j==1:
                second_eth_images_and_labels_df_train = second_eth_images_and_labels_df_train.append(pd.DataFrame(data={'image': id_images_train, 'label': i}))
                second_eth_images_and_labels_df_test = second_eth_images_and_labels_df_test.append(pd.DataFrame(data={'image': id_images_test, 'label': i}))
            i += 1

    first_eth_images_and_labels_df_train = first_eth_images_and_labels_df_train.sample(frac=1).reset_index(drop=True)  # shuffle table
    first_eth_images_and_labels_df_test = first_eth_images_and_labels_df_test.sample(frac=1).reset_index(drop=True)  # shuffle table
    second_eth_images_and_labels_df_train = second_eth_images_and_labels_df_train.sample(frac=1).reset_index(drop=True)  # shuffle table
    second_eth_images_and_labels_df_test = second_eth_images_and_labels_df_test.sample(frac=1).reset_index(drop=True)  # shuffle table

    transform_train = transforms.Compose(
        [transforms.Resize([256, 256]), transforms.RandomCrop([224, 224]), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
         
    transform_test = transforms.Compose(
        [transforms.Resize([256, 256]), transforms.CenterCrop([224, 224]), transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
         
    first_train_dataset = MyDataset(images_path, first_eth_images_and_labels_df_train, transform_train)
    first_test_dataset = MyDataset(images_path, first_eth_images_and_labels_df_test, transform_test)
    second_train_dataset = MyDataset(images_path, second_eth_images_and_labels_df_train, transform_train)
    second_test_dataset = MyDataset(images_path, second_eth_images_and_labels_df_test, transform_test)
    
    first_train_dataloader = DataLoader(first_train_dataset, batch_size=64, shuffle=True, num_workers=4)
    first_test_dataloader = DataLoader(first_test_dataset, batch_size=64, shuffle=True, num_workers=4)
    second_train_dataloader = DataLoader(second_train_dataset, batch_size=64, shuffle=True, num_workers=4)
    second_test_dataloader = DataLoader(second_test_dataset, batch_size=64, shuffle=True, num_workers=4)

    # train and test model
    obj = LocalModelStore("DoubleTailVgg16", f'{split_layer}_split', model_path)
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_epoch(first_train_dataloader, second_train_dataloader, model, loss_fn, head_optimizer, tail_optimizer)
        tail_stepLR.step()
        head_stepLR.step()
        first_acc, first_loss = test_epoch(first_test_dataloader, model, loss_fn, "first")
        second_acc, second_loss = test_epoch(second_test_dataloader, model, loss_fn, "second")
        print(f"Overall Test Error: \n Accuracy: {np.mean([first_acc, second_acc]):>0.1f}%, Avg loss: {np.mean([first_loss, second_loss]):>8f} \n")
       
        # save model
        if epoch % 10 == 0:
            obj.save_model(model, head_optimizer, tail_optimizer, epoch, np.mean([first_acc, second_acc]), False)

    first_acc, first_loss = test_epoch(first_test_dataloader, model, loss_fn, "first")
    second_acc, second_loss = test_epoch(second_test_dataloader, model, loss_fn, "second")
    print(f"Overall Test Error: \n Accuracy: {np.mean([first_acc, second_acc]):>0.1f}%, Avg loss: {np.mean([first_loss, second_loss]):>8f} \n")
    obj.save_model(model, head_optimizer, tail_optimizer, epoch, np.mean([first_acc, second_acc]), False)

    print("-------------------------------------------")
    print("Done with training! Strat Verification test:")
   
    # runs verification test for each test type in the dict
    for test_type in list(test_types_mapped_to_pairs_file_name_dict.keys()):
        verification_test(model, dir_pairs_path, test_types_mapped_to_pairs_file_name_dict[test_type], dataset_path, test_type)
        print(f'Done {test_type} test type')
    
    print("-------------------------------------------")
    print("finised running.")



images_path = "/home/ssd_storage/datasets/students/OM/datasets/mixed_for_DoubleTail/"
model_path = "/home/ssd_storage/experiments/students/OM/doubleTailExp/"
eths = ["2", "3"]
n_classes_first_race = 477
n_classes_second_race = 477
split_layer = "conv2"
epochs = 50
train_perc = 0.8
dir_pairs_path = "/home/ssd_storage/datasets/students/OM/"
test_types_mapped_to_pairs_file_name_dict = {"same_first_eth": "same_2.txt", "diff_first_eth": "diff_2.txt", "same_second_eth": "same_3.txt", "diff_second_eth": "diff_3.txt"}
dataset_path = "/home/administrator/datasets/vggface2_mtcnn/"
seed = 42
    
train_DoubleTailVgg16_model(images_path, model_path, eths, n_classes_first_race, n_classes_second_race, split_layer, epochs, train_perc, dir_pairs_path, test_types_mapped_to_pairs_file_name_dict, dataset_path, seed)
