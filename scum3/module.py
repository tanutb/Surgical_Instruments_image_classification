import torch
from torch.autograd import Variable
from tqdm import tqdm
import time

def testAccuracy(model,loss_fn,dataset):
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    running_loss = 0.0
    
    with torch.no_grad():
        for data in dataset:
            images, labels = data
            images, labels = images.cuda() , labels.cuda()
            # run the model on the test set to predict labels
            model.cuda()
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
            loss = loss_fn(outputs, labels)
            running_loss += loss.item() 
            
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return accuracy , running_loss/ len(dataset)


def train(num_epochs,model,optimizer,loss_fn , traindata , validationdata , name = "Best_model.pt"):

    best_val_loss = float("inf")
    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", torch.cuda.get_device_name(device), "device")
    # Convert model parameters and buffers to CPU or Cuda
    start_time = time.time()

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0
        pbar = tqdm(total=len(traindata), desc=f"Epoch {epoch+1}/{num_epochs}", position=0, leave=True)
        for i, (images, labels) in enumerate(traindata, 0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            model.to(device)
            # print(images.dim())
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            pbar.update(1)
            pbar.set_postfix({'Loss': running_loss / len(traindata)})
        pbar.close()
        accuracy , loss  = testAccuracy(model,loss_fn,validationdata)
        print('For epoch {}, the test accuracy over the whole validation set is {}% and loss {}'.format(epoch+1, accuracy , loss))
        if loss < best_val_loss:
            best_val_loss = loss
            torch.save(model.state_dict(), name)
    end_time = time.time()
    print("Training complete!")
    print("Total training time: {:.2f} seconds".format(end_time - start_time))