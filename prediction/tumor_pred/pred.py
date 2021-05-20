import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from torch.autograd import Variable
import time
import torch.nn.parallel
import sys
import torch.nn.functional as F

from model_paad import PreActResNet34

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


APS = 350
PS = 224
TileFolder = sys.argv[1] + "/"

BatchSize = 96

heat_map_out = sys.argv[3]
old_model = sys.argv[4]

mu = [0.7238, 0.5716, 0.6779]
sigma = [0.1120, 0.1459, 0.1089]


device = "cuda" if torch.cuda.is_available() else "cpu"
data_aug = transforms.Compose(
    [transforms.Scale(PS), transforms.ToTensor(), transforms.Normalize(mu, sigma)]
)


def whiteness(png):
    wh = (
        np.std(png[:, :, 0].flatten())
        + np.std(png[:, :, 1].flatten())
        + np.std(png[:, :, 2].flatten())
    ) / 3.0
    return wh


def softmax_np(x):
    x = x - np.max(x, 1, keepdims=True)
    x = np.exp(x) / (np.sum(np.exp(x), 1, keepdims=True))
    return x


def iterate_minibatches(inputs, augs, targets):
    if inputs.shape[0] <= BatchSize:
        yield inputs, augs, targets
        return

    start_idx = 0
    for start_idx in range(0, len(inputs) - BatchSize + 1, BatchSize):
        excerpt = slice(start_idx, start_idx + BatchSize)
        yield inputs[excerpt], augs[excerpt], targets[excerpt]
    if start_idx < len(inputs) - BatchSize:
        excerpt = slice(start_idx + BatchSize, len(inputs))
        yield inputs[excerpt], augs[excerpt], targets[excerpt]


def load_data(todo_list, rind):
    X = torch.zeros(size=(BatchSize * 40, 3, PS, PS))
    inds = np.zeros(shape=(BatchSize * 40,), dtype=np.int32)
    coor = np.zeros(shape=(200000, 2), dtype=np.int32)

    # TODO: the comment below doesn't seem to be correct...
    # change this to true if dont have images normalized and normalize on the fly
    normalized = False
    parts = 4
    if normalized:
        parts = 4

    xind = 0
    lind = 0
    cind = 0
    for fn in todo_list:
        lind += 1
        full_fn = TileFolder + "/" + fn
        if not os.path.isfile(full_fn):
            continue
        if (len(fn.split("_")) != parts) or (".png" not in fn):
            continue

        try:
            x_off = float(fn.split("_")[0])
            y_off = float(fn.split("_")[1])
            svs_pw = float(fn.split("_")[2])
            png_pw = float(fn.split("_")[3].split(".png")[0])
        except:
            print("error reading image")
            continue

        png = np.array(Image.open(full_fn).convert("RGB"))
        for x in range(0, png.shape[1], APS):
            if x + APS > png.shape[1]:
                continue
            for y in range(0, png.shape[0], APS):
                if y + APS > png.shape[0]:
                    continue

                if whiteness(png[y : y + APS, x : x + APS, :]) >= 12:
                    a = png[y : y + APS, x : x + APS, :]
                    a = Image.fromarray(a.astype("uint8"), "RGB")
                    a = data_aug(a)
                    X[xind, :, :, :] = a
                    inds[xind] = rind
                    xind += 1

                coor[cind, 0] = np.int32(x_off + (x + APS / 2) * svs_pw / png_pw)
                coor[cind, 1] = np.int32(y_off + (y + APS / 2) * svs_pw / png_pw)

                cind += 1
                rind += 1
                if rind % 100 == 0:
                    print("Processed: ", rind)
        if xind >= BatchSize:
            break

    X = X[0:xind]
    inds = inds[0:xind]
    coor = coor[0:cind]

    return todo_list[lind:], X, inds, coor, rind


def val_fn_epoch_on_disk(classn, val_fn):
    all_or = np.zeros(shape=(500000, classn), dtype=np.float32)
    all_inds = np.zeros(shape=(500000,), dtype=np.int32)
    all_coor = np.zeros(shape=(500000, 2), dtype=np.int32)
    rind = 0
    n1 = 0
    n2 = 0
    n3 = 0
    todo_list = os.listdir(TileFolder)
    processed = 0
    total = len(todo_list)
    start = time.time()
    coor_c = 0
    while len(todo_list) > 0:
        todo_list, inputs, inds, coor, rind = load_data(todo_list, rind)
        coor_c += len(coor)

        # if len(inputs) == 0:
        #    print('len of inputs is 0"')
        #    break;
        if inputs.size(0) < 2:
            print("len of inputs if less than 2")
        else:
            processed = total - len(todo_list)
            print(
                "Processed: {}/{} \t Time Remaining: {}mins".format(
                    processed,
                    total,
                    (time.time() - start) / 60 * (total / processed - 1),
                )
            )
            with torch.no_grad():
                inputs = Variable(inputs.to(device))
                output = val_fn(inputs)

            output = F.sigmoid(output)
            output = output.data.cpu().numpy()    # TODO: the comment below doesn't seem to be correct...

            print("size of output: ", output.shape)

            # output = softmax_np(output)[:, 1]
            all_or[n1 : n1 + len(output)] = output.reshape(-1, 1)
            n1 += len(output)
            all_inds[n2 : n2 + len(inds)] = inds
            n2 += len(inds)

        all_coor[n3 : n3 + len(coor)] = coor
        n3 += len(coor)

    all_or = all_or[:n1]
    all_inds = all_inds[:n2]
    all_coor = all_coor[:n3]
    return all_or, all_inds, all_coor


# load model
print("start predicting...")
start = time.time()

print("| Load pretrained at  %s..." % old_model)

checkpoint = torch.load(old_model, map_location=device)
model = PreActResNet34(1)
model.load_state_dict(checkpoint["net"])
model.to(device)
model.eval()
best_auc = checkpoint["acc"]
print("previous best AUC: {:.4f} at epoch: {}".format(best_auc, checkpoint["epoch"]))
print("=============================================")


Or, inds, coor = val_fn_epoch_on_disk(1, model)
Or_all = np.zeros(shape=(coor.shape[0],), dtype=np.float32)
Or_all[inds] = Or[:, 0]

print("len of all coor: ", coor.shape)
print("shape of Or: ", Or.shape)
print("shape of inds: ", inds.shape)

fid = open(TileFolder + "/" + heat_map_out, "w")
for idx in range(0, Or_all.shape[0]):
    fid.write("{} {} {}\n".format(coor[idx][0], coor[idx][1], Or_all[idx]))

fid.close()

print("Elapsed Time: ", (time.time() - start) / 60.0)
print("DONE!")
