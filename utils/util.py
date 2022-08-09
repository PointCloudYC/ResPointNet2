import sys
import os
import itertools
from string import ascii_uppercase
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib as mpl
# mpl.rc('axes', labelsize=14)
# mpl.rc('xtick', labelsize=12)
# mpl.rc('ytick', labelsize=12)
import seaborn as sns


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def classification_metrics(preds, targets, num_classes):
    seen_class = [0.0 for _ in range(num_classes)]
    correct_class = [0.0 for _ in range(num_classes)]
    preds = np.argmax(preds, -1)
    correct = np.sum(preds == targets)
    seen = preds.shape[0]
    for l in range(num_classes):
        seen_class[l] += np.sum(targets == l)
        correct_class[l] += (np.sum((preds == l) & (targets == l)))
    acc = 1.0 * correct / seen
    avg_class_acc = np.mean(np.array(correct_class) / np.array(seen_class))
    return acc, avg_class_acc


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def partnet_metrics(num_classes, num_parts, objects, preds, targets):
    """

    Args:
        num_classes:
        num_parts:
        objects: [int]
        preds:[(num_parts,num_points)]
        targets: [(num_points)]

    Returns:

    """
    shape_iou_tot = [0.0] * num_classes
    shape_iou_cnt = [0] * num_classes
    part_intersect = [np.zeros((num_parts[o_l]), dtype=np.float32) for o_l in range(num_classes)]
    part_union = [np.zeros((num_parts[o_l]), dtype=np.float32) + 1e-6 for o_l in range(num_classes)]

    for obj, cur_pred, cur_gt in zip(objects, preds, targets):
        cur_num_parts = num_parts[obj]
        cur_pred = np.argmax(cur_pred[1:, :], axis=0) + 1
        cur_pred[cur_gt == 0] = 0
        cur_shape_iou_tot = 0.0
        cur_shape_iou_cnt = 0
        for j in range(1, cur_num_parts):
            cur_gt_mask = (cur_gt == j)
            cur_pred_mask = (cur_pred == j)

            has_gt = (np.sum(cur_gt_mask) > 0)
            has_pred = (np.sum(cur_pred_mask) > 0)

            if has_gt or has_pred:
                intersect = np.sum(cur_gt_mask & cur_pred_mask)
                union = np.sum(cur_gt_mask | cur_pred_mask)
                iou = intersect / union

                cur_shape_iou_tot += iou
                cur_shape_iou_cnt += 1

                part_intersect[obj][j] += intersect
                part_union[obj][j] += union
        if cur_shape_iou_cnt > 0:
            cur_shape_miou = cur_shape_iou_tot / cur_shape_iou_cnt
            shape_iou_tot[obj] += cur_shape_miou
            shape_iou_cnt[obj] += 1

    msIoU = [shape_iou_tot[o_l] / shape_iou_cnt[o_l] for o_l in range(num_classes)]
    part_iou = [np.divide(part_intersect[o_l][1:], part_union[o_l][1:]) for o_l in range(num_classes)]
    mpIoU = [np.mean(part_iou[o_l]) for o_l in range(num_classes)]

    # Print instance mean
    mmsIoU = np.mean(np.array(msIoU))
    mmpIoU = np.mean(mpIoU)

    return msIoU, mpIoU, mmsIoU, mmpIoU


def IoU_from_confusions(confusions):
    """
    Computes IoU from confusion matrices.
    :param confusions: ([..., n_c, n_c] np.int32). Can be any dimension, the confusion matrices should be described by
    the last axes. n_c = number of classes
    :param ignore_unclassified: (bool). True if the the first class should be ignored in the results
    :return: ([..., n_c] np.float32) IoU score
    """

    # Compute TP, FP, FN. This assume that the second to last axis counts the truths (like the first axis of a
    # confusion matrix), and that the last axis counts the predictions (like the second axis of a confusion matrix)
    TP = np.diagonal(confusions, axis1=-2, axis2=-1)
    TP_plus_FN = np.sum(confusions, axis=-1)
    TP_plus_FP = np.sum(confusions, axis=-2)

    # Compute IoU
    IoU = TP / (TP_plus_FP + TP_plus_FN - TP + 1e-6)

    # Compute mIoU with only the actual classes
    mask = TP_plus_FN < 1e-3
    counts = np.sum(1 - mask, axis=-1, keepdims=True)
    mIoU = np.sum(IoU, axis=-1, keepdims=True) / (counts + 1e-6)

    # If class is absent, place mIoU in place of 0 IoU to get the actual mean later
    IoU += mask * mIoU

    return IoU


def seg_metrics(num_classes, vote_logits, validation_proj, validation_labels):
    Confs = []
    for logits, proj, targets in zip(vote_logits, validation_proj, validation_labels):
        preds = np.argmax(logits[:, proj], axis=0).astype(np.int32)
        Confs += [confusion_matrix(targets, preds, np.arange(num_classes))]
    # Regroup confusions
    C = np.sum(np.stack(Confs), axis=0)

    IoUs = IoU_from_confusions(C)
    mIoU = np.mean(IoUs)
    return IoUs, mIoU


def seg_metrics_vis_CM(num_classes, vote_logits, 
                         validation_proj, validation_labels, 
                         more_metrics=False, image_path=None,
                         label_to_names=None,visualize_CM=True):
    """compute metrics (miou, OA, precision, recall), visualize confusion matrix
    Args:
        num_classes ([type]): the number of class
        vote_logits ([type]): vote_logits for all sub points
        validation_proj ([type]): validation projection, i.e. the nearest nb sub-pt indice for each original pt
        validation_labels ([type]): validation labels for each original pt
        more_metrics (bool, optional): compute other metrics (precision, recall, OA,etc). Defaults to False.
        image_path ([type], optional): place to save CM picture. Defaults to None.
        label_to_names ([type], optional): label and name dictionary for the dataset. Defaults to None.
        visualize_CM (bool, optional): show the CM. Defaults to True.
    Returns:
        [type]: [description]
    """

    # compute mIoU
    Confs = []
    targets_list=[]
    preds_list=[]
    for logits, proj, targets in zip(vote_logits, validation_proj, validation_labels): # vote_logits for sub-pc, validation_{proj,labels} for pc.
        preds = np.argmax(logits[:, proj], axis=0).astype(np.int32) # amazing, get preds based on sub-pc's preds(logits var) for all original pts, logits shape Cx#num_subpc_pts
        Confs += [confusion_matrix(targets, preds, np.arange(num_classes))] # confusion_matrix is just a np array
        targets_list.append(targets)
        preds_list.append(preds)
    # Regroup confusions
    C = np.sum(np.stack(Confs), axis=0)
    IoUs = IoU_from_confusions(C)
    mIoU = np.mean(IoUs)
        
    # compute other metrics
    if more_metrics:
        y_true=np.vstack(targets_list).squeeze()
        y_pred=np.vstack(preds_list).squeeze() 
        target_names = list(label_to_names.values())
        cls_report = classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                digits=4)
        print(cls_report)

        # overall_acc1, _ = classification_metrics(y_pred,y_true,num_classes)
        cls_report_dict = classification_report(
                y_true,
                y_pred,
                target_names=target_names,
                digits=4,output_dict=True)
        overall_acc = cls_report_dict['accuracy']
        # print('Overall accuracy is {:.4f} or {:.4f}'.format(overall_acc,overall_acc1))
    else:
        print('Attention, overall accuracy is not computed.')
        overall_acc = None

    # plot CM
    if visualize_CM:
        print(f"save confusion matrix at root{image_path}")
        plot_CM_wrapper(C,y_true,y_pred,label_to_names,image_path,filename='CM_seaborn')

    return IoUs, mIoU, overall_acc


def plot_CM_wrapper(cm,y_true,y_pred,label_to_names,image_path,filename='CM_seaborn'):

    assert label_to_names is not None
    label_values = np.sort([k for k, v in label_to_names.items()])
    target_names = list(label_to_names.values())

    # plot w. scikit-learn style
    plot_confusion_matrix2(
                    cm, 
                    image_path,
                    normalize    = False,
                    target_names = target_names,
                    title        = "Confusion Matrix")

    # plot w. seaborn style
    plot_confusion_matrix_seaborn(
        y_true,
        y_pred,
        filename,
        label_values,
        label_to_names,
        image_path, 
        figsize=(10,10))    


def header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append('element vertex %d' % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append('property %s %s' % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the 
        file name if it does no already have one.

    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a 
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered 
        as one field. 

    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of 
        fields.

    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])

    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])

    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)

    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print('fields have more than 2 dimensions')
            return False    

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print('wrong field dimensions')
        return False    

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if (n_fields != len(field_names)):
        print('wrong number of field names')
        return False

    # Add extension if not there
    if not filename.endswith('.ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as plyfile:

        # First magical word
        header = ['ply']

        # Encoding format
        header.append('format binary_' + sys.byteorder + '_endian 1.0')

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append('element face {:d}'.format(triangular_faces.shape[0]))
            header.append('property list uchar int vertex_indices')

        # End of header
        header.append('end_header')

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, 'ab') as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [('k', 'uint8')] + [(str(ind), 'int32') for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data['k'] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data['0'] = triangular_faces[:, 0]
            data['1'] = triangular_faces[:, 1]
            data['2'] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True


def save_predicted_results_PLY(vote_logits, validation_proj, 
                               validation_points,validation_colors, 
                               validation_labels,test_path=None,
                               cloud_names=None, open3d_visualize=False):
    """save predicted results as ply for each area
    """
    os.makedirs(os.path.join(test_path, 'results'),exist_ok=True)
    for logits, proj, points, colors, targets, cloud_name in zip(vote_logits, validation_proj,validation_points,validation_colors, validation_labels,cloud_names):
        preds = np.argmax(logits[:, proj], axis=0).astype(np.int32)
        pred_name = os.path.join(test_path, 'results', f'{cloud_name}_pred.ply')
        gt_name = os.path.join(test_path, 'results', f'{cloud_name}_gt.ply')

        write_ply(pred_name,
                [points, preds],
                ['x', 'y', 'z', 'preds'])
        print(f"{cloud_name}_pred.ply saved successfully")
        write_ply(gt_name,
                [points, targets],
                ['x', 'y', 'z', 'gt'])
        print(f"{cloud_name}_gt.ply saved successfully")

        if open3d_visualize:
            from .visualize import Plot
            xyzrgb = np.concatenate([points, colors], axis=-1)
            Plot.draw_pc(xyzrgb)  # visualize raw point clouds
            Plot.draw_pc_sem_ins(points, targets)  # visualize ground-truth
            Plot.draw_pc_sem_ins(points, preds)  # visualize prediction


def sub_seg_metrics(num_classes, validation_logits, validation_labels, val_proportions):
    Confs = []
    for logits, targets in zip(validation_logits, validation_labels):
        preds = np.argmax(logits, axis=0).astype(np.int32)
        Confs += [confusion_matrix(targets, preds, np.arange(num_classes))]
    # Regroup confusions
    C = np.sum(np.stack(Confs), axis=0).astype(np.float32)
    # Rescale with the right number of point per class
    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)
    IoUs = IoU_from_confusions(C)
    mIoU = np.mean(IoUs)

    return IoUs, mIoU


def seg_part_metrics(num_classes, predictions, targets, val_proportions):
    # Confusions for subparts of validation set
    Confs = np.zeros((len(predictions), num_classes, num_classes), dtype=np.int32)
    for i, (probs, truth) in enumerate(zip(predictions, targets)):
        # Predicted labels
        preds = np.argmax(probs, axis=0)
        # Confusions
        Confs[i, :, :] = confusion_matrix(truth, preds, np.arange(num_classes))
    # Sum all confusions
    C = np.sum(Confs, axis=0).astype(np.float32)
    # Balance with real validation proportions
    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)
    # Objects IoU
    IoUs = IoU_from_confusions(C)
    # Print instance mean
    mIoU = np.mean(IoUs)
    return IoUs, mIoU


def shapenetpart_metrics(num_classes, num_parts, objects, preds, targets, masks):
    """
    Args:
        num_classes:
        num_parts:
        objects: [int]
        preds:[(num_parts,num_points)]
        targets: [(num_points)]
        masks: [(num_points)]
    """
    total_correct = 0.0
    total_seen = 0.0
    Confs = []
    for obj, cur_pred, cur_gt, cur_mask in zip(objects, preds, targets, masks):
        obj = int(obj)
        cur_num_parts = num_parts[obj]
        cur_pred = np.argmax(cur_pred, axis=0)
        cur_pred = cur_pred[cur_mask]
        cur_gt = cur_gt[cur_mask]
        correct = np.sum(cur_pred == cur_gt)
        total_correct += correct
        total_seen += cur_pred.shape[0]
        parts = [j for j in range(cur_num_parts)]
        Confs += [confusion_matrix(cur_gt, cur_pred, labels=parts)]

    Confs = np.array(Confs)
    obj_mIoUs = []
    objects = np.asarray(objects)
    for l in range(num_classes):
        obj_inds = np.where(objects == l)[0]
        obj_confs = np.stack(Confs[obj_inds])
        obj_IoUs = IoU_from_confusions(obj_confs)
        obj_mIoUs += [np.mean(obj_IoUs, axis=-1)]

    objs_average = [np.mean(mIoUs) for mIoUs in obj_mIoUs]
    instance_average = np.mean(np.hstack(obj_mIoUs))
    class_average = np.mean(objs_average)
    acc = total_correct / total_seen

    print('Objs | Inst | Air  Bag  Cap  Car  Cha  Ear  Gui  Kni  Lam  Lap  Mot  Mug  Pis  Roc  Ska  Tab')
    print('-----|------|--------------------------------------------------------------------------------')

    s = '{:4.1f} | {:4.1f} | '.format(100 * class_average, 100 * instance_average)
    for AmIoU in objs_average:
        s += '{:4.1f} '.format(100 * AmIoU)
    print(s + '\n')
    return acc, objs_average, class_average, instance_average


def save_fig(image_path, fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(image_path, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)


# a btter CM plot function (July 26, 2020)
# ref: https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
def plot_confusion_matrix2(cm,image_path,filename='CM_default',
                          target_names=None,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('true label')
    plt.xlabel('predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    save_fig(image_path,filename, tight_layout=False)
    plt.show()


# TODO: need to optimize the speed, can refer to the cm_analysis2 function
def plot_confusion_matrix_seaborn(y_true, y_pred, filename, labels,label_to_names,image_path, ymap=None, figsize=(10,10), normalized=False):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    if normalized:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100 # 100% form
    cm = pd.DataFrame(cm, index=label_to_names.values(), columns=label_to_names.values())
    cm.index.name = 'true labels'
    cm.columns.name = 'predicted labels'
    fig, ax = plt.subplots(figsize=figsize)

    # fmt =',d' to show numeric number only, e.g., 1,200,000
    # https://stackoverflow.com/questions/59990375/plot-confusioin-matrix-plot-is-not-showing-integer-value-instead-it-is-showing-s
    if normalized:
        sns.heatmap(cm, cmap='Oranges', annot=True,ax=ax, fmt='0.2f')
    else:
        sns.heatmap(cm, cmap='Oranges', annot=True,ax=ax, fmt=',d')
    save_fig(image_path,filename, tight_layout=False)

def cm_analysis2(y_true, y_pred, filename, labels,label_to_names,image_path, ymap=None, figsize=(10,10)):

    columns = ['class %s' %(i) for i in list(ascii_uppercase)[0:len(np.unique(y_true))]]

    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cm, index=columns, columns=columns)

    ax = sns.heatmap(df_cm, cmap='Oranges', annot=True)
    plt.show()
    save_fig(image_path,filename, tight_layout=False)


# Define PLY types
ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

# Numpy reader format
valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}

def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        if b'element' in line:
            line = line.split()
            num_points = int(line[2])

        elif b'property' in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None


    while b'end_header' not in line and line != b'':
        line = plyfile.readline()

        # Find point element
        if b'element vertex' in line:
            current_element = 'vertex'
            line = line.split()
            num_points = int(line[2])

        elif b'element face' in line:
            current_element = 'face'
            line = line.split()
            num_faces = int(line[2])

        elif b'property' in line:
            if current_element == 'vertex':
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == 'vertex':
                if not line.startswith('property list uchar int'):
                    raise ValueError('Unsupported faces property : ' + line)

    return num_points, num_faces, vertex_properties

def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files

    Parameters
    ----------
    filename : string
        the name of the file to read.

    Returns
    -------
    result : array
        data stored in the file

    Examples
    --------
    Store data in file

    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])

    Read the file

    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])
    
    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])

    """

    with open(filename, 'rb') as plyfile:


        # Check if the file start with ply
        if b'ply' not in plyfile.readline():
            raise ValueError('The file does not start whith the word ply')

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('The file is not binary')

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [('k', ext + 'u1'),
                               ('v1', ext + 'i4'),
                               ('v2', ext + 'i4'),
                               ('v3', ext + 'i4')]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data['v1'], faces_data['v2'], faces_data['v3'])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data
