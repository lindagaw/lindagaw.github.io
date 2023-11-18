import numpy as np
import torch
import torch.nn as nn
from datasets import Array3D, ClassLabel, Features, load_dataset
from matplotlib import pyplot
from numpy import inf
from sklearn.utils.class_weight import compute_class_weight
from torchinfo import summary
from tqdm import tqdm
from transformers import AdamW, ViTFeatureExtractor, ViTModel

# load cifar10
train_ds, test_ds = load_dataset('cifar10', split=['train[:5000]', 'test[:2000]'])
# split up training into training + validation
splits = train_ds.train_test_split(test_size=0.1)
train_ds = splits['train']
val_ds = splits['test']
chk_img = train_ds[67]
img = np.array(chk_img['img'])
print("Shape: ", img.shape)
cats = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print("Animal: ", cats[chk_img['label']])
pyplot.imshow(img, cmap=pyplot.get_cmap('gray'))

def preprocess_images(examples):
    # get batch of images
    images = examples['img']
    # convert to list of NumPy arrays of shape (C, H, W)
    images = [np.array(image, dtype=np.uint8) for image in images]
    images = [np.moveaxis(image, source=-1, destination=0) for image    in images]
    # preprocess and add pixel_values
    inputs = feature_extractor(images=images)
    examples['pixel_values'] = inputs['pixel_values']
    return examples

features = Features({
    'label': ClassLabel(names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']),
    'img': Array3D(dtype="int64", shape=(3,32,32)),
    'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
})
preprocessed_train_ds = train_ds.map(preprocess_images, batched=True, features=features)
preprocessed_val_ds = val_ds.map(preprocess_images, batched=True, features=features)
preprocessed_test_ds = test_ds.map(preprocess_images, batched=True, features=features)
# set format to PyTorch
preprocessed_train_ds.set_format('torch', columns=['pixel_values', 'label'])
preprocessed_val_ds.set_format('torch', columns=['pixel_values', 'label'])
preprocessed_test_ds.set_format('torch', columns=['pixel_values', 'label'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# create dataloaders
train_batch_size = 10
eval_batch_size = 10
train_dataloader = torch.utils.data.DataLoader(preprocessed_train_ds, batch_size=train_batch_size, shuffle=True, num_workers=2)
val_dataloader = torch.utils.data.DataLoader(preprocessed_val_ds, batch_size=eval_batch_size, num_workers=2)
test_dataloader = torch.utils.data.DataLoader(preprocessed_test_ds, batch_size=eval_batch_size, num_workers=2)
batch = next(iter(train_dataloader))

assert batch['pixel_values'].shape == (train_batch_size, 3, 224, 224)
assert batch['label'].shape == (train_batch_size,)

vit_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')

for param in vit_model.parameters():
      param.requires_grad = False

class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=10):
        super(ViTForImageClassification, self).__init__()
        self.vit = vit_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size,    num_labels)
        self.num_labels = num_labels
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)
        return logits
model = ViTForImageClassification()
model = model.to(device)

#compute the class weights
class_wts = compute_class_weight("balanced", np.unique(preprocessed_train_ds['label']), preprocessed_train_ds['label'].tolist())
# print(class_wts)
# # convert class weights to tensor
weights= torch.tensor(class_wts,dtype=torch.float)
weights = weights.to(device)
# # loss function
cross_entropy = nn.CrossEntropyLoss(weight=weights)

# define the optimizer
optimizer = AdamW(model.parameters(), lr = 1e-3)
# number of training epochs
epochs = 50

# function to train the model
def train():

    model.train()
    total_loss = 0
    # empty list to save model predictions
    total_preds=[]
    # iterate over batches
    for step,batch in enumerate(train_dataloader):
        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step,    len(train_dataloader)))

        # push the batch to gpu
        lbl, pix = batch.items()
        lbl, pix = lbl[1].to(device), pix[1].to(device)

        # get model predictions for the current batch
        preds = model(pix)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, lbl)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the    exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # clear calculated gradients
        optimizer.zero_grad()
        preds=preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)
    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds

def eval():
total_loss = 0
model.eval() # prep model for evaluation
for step,batch in enumerate(val_dataloader):
    lbl, pix = batch.items()
    lbl, pix = lbl[1].to(device), pix[1].to(device)

    # forward pass: compute predicted outputs by passing inputs to the model
    preds = model(pix)
    # calculate the loss
    loss = cross_entropy(preds, lbl)
    total_loss += loss.item()

return total_loss / len(val_dataloader)

min_loss = inf
es = 0
for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))

    # Train model
    train_loss, _ = train()
    val_loss = eval()

    # Early Stopping
    if val_loss < min_loss:
        min_loss = val_loss
        es = 0
    else:
        es += 1
        if es > 4:
            print("Early stopping with train_loss: ", train_loss, "and val_loss for this epoch: ", val_loss, "...")
            break

    # it can make your experiment reproducible, similar to set  random seed to all options where there needs a random seed.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f'\n Training Loss: {train_loss:.3f}')
    print(f'\n Validation Loss: {val_loss:.3f}')


torch.save(model.state_dict(), '/working/model')

def eval():
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for step, batch in tqdm(enumerate(test_dataloader), total = len(test_dataloader)):
            lbl, pix = batch.items()
            lbl, pix = lbl[1].to(device), pix[1].to(device)
            outputs = model(pix)
            outputs = torch.argmax(outputs, axis=1)
            y_pred.extend(outputs.cpu().detach().numpy())
            y_true.extend(lbl.cpu().detach().numpy())

    return y_pred, y_true
y_pred, y_true = eval()

correct = np.array(y_pred) == np.array(y_true)
accuracy = correct.sum() / len(correct)
print("Accuracy of the model", accuracy)
