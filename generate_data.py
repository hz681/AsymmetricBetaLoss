import os
import numpy as np


def load_data(base_path):
    data = {}
    for phase in ['train', 'val']:
        data[phase] = {}
        data[phase]['images'] = np.load(os.path.join(base_path, 'formatted_{}_images.npy'.format(phase)))
        data[phase]['labels'] = np.load(os.path.join(base_path, 'formatted_{}_labels.npy'.format(phase)))
    return data


def load_imagenet21k(path):
    data = []
    class_list = os.listdir(path)
    for cls in class_list:
        p = os.path.join(path, cls)
        imagelist = os.listdir(p)
        imagelist = list(map(lambda x: os.path.join(p, x), imagelist))
        data.append(imagelist)
    data = np.concatenate(data)
    data = np.sort(data)
    return data


# without openset data
def gen_data_without_openset(dataset_dir, dataset_name, seed, save=True):

    nrs = np.random.RandomState(seed)
    close_set_data = load_data(os.path.join(dataset_dir, dataset_name))
    if dataset_name == 'voc':
        close_set_data['train']['images'] = np.char.add(
            f'{dataset_dir}/voc/VOCdevkit/VOC2012/JPEGImages/', close_set_data['train']['images'])
        close_set_data['val']['images'] = np.char.add(
            f'{dataset_dir}/voc/VOCdevkit/VOC2012/JPEGImages/', close_set_data['val']['images'])
    if dataset_name == 'coco':
        close_set_data['train']['images'] = np.char.add(
            f'{dataset_dir}/coco/', close_set_data['train']['images'])
        close_set_data['val']['images'] = np.char.add(
            f'{dataset_dir}/coco/', close_set_data['val']['images'])
    if dataset_name == 'nus':
        close_set_data['train']['images'] = np.char.add(
            f'{dataset_dir}/nus/Flickr/', close_set_data['train']['images'])
        close_set_data['val']['images'] = np.char.add(
            f'{dataset_dir}/nus/Flickr/', close_set_data['val']['images'])
    
    n_train = len(close_set_data['train']['labels'])
    indices = nrs.permutation(n_train)

    for lb_ratio in [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]:

        n_lb = int(lb_ratio * n_train)
        lb_idxs = indices[:n_lb]
        ub_idxs = indices[n_lb:]
        lb_train_imgs = close_set_data['train']['images'][lb_idxs]
        lb_train_labels = close_set_data['train']['labels'][lb_idxs]
        ub_train_imgs = close_set_data['train']['images'][ub_idxs]
        ub_train_labels = close_set_data['train']['labels'][ub_idxs]

        idx = nrs.permutation(len(ub_train_labels))
        ub_train_imgs = ub_train_imgs[idx]
        ub_train_labels = ub_train_labels[idx]

        print('generate data success!')
        print(f'len(labeled data) {len(lb_idxs)}')
        print(f'len(unlabeled close data) {len(ub_idxs)}')
        print(f"len(test data) {len(close_set_data['val']['labels'])}")
    
        if save:
            path = f'./data/{dataset_name}/{seed}'
            os.makedirs(path, exist_ok=True)
            np.save(f'{path}/lb_image_{lb_ratio}.npy', lb_train_imgs)
            np.save(f'{path}/lb_label_{lb_ratio}.npy', lb_train_labels)
            np.save(f'{path}/ub_image_{lb_ratio}.npy', ub_train_imgs)
            np.save(f'{path}/ub_label_{lb_ratio}.npy', ub_train_labels)
            np.save(f'{path}/test_image.npy', close_set_data['val']['images'])
            np.save(f'{path}/test_label.npy', close_set_data['val']['labels'])


# use imagenet21k as open set
def gen_data_with_in21k(dataset_dir, dataset_name, seed, save=True):

    nrs = np.random.RandomState(seed)
    close_set_data = load_data(os.path.join(dataset_dir, dataset_name))
    if dataset_name == 'voc':
        close_set_data['train']['images'] = np.char.add(
            f'{dataset_dir}/voc/VOCdevkit/VOC2012/JPEGImages/', close_set_data['train']['images'])
        close_set_data['val']['images'] = np.char.add(
            f'{dataset_dir}/voc/VOCdevkit/VOC2012/JPEGImages/', close_set_data['val']['images'])
    if dataset_name == 'coco':
        close_set_data['train']['images'] = np.char.add(
            f'{dataset_dir}/coco/', close_set_data['train']['images'])
        close_set_data['val']['images'] = np.char.add(
            f'{dataset_dir}/coco/', close_set_data['val']['images'])
    if dataset_name == 'nus':
        close_set_data['train']['images'] = np.char.add(
            f'{dataset_dir}/nus/Flickr/', close_set_data['train']['images'])
        close_set_data['val']['images'] = np.char.add(
            f'{dataset_dir}/nus/Flickr/', close_set_data['val']['images'])
    
    if dataset_name == 'voc' or dataset_name == 'coco':
        open_set_data = load_imagenet21k(f'{dataset_dir}/imagenet21k_select1/')
    if dataset_name == 'nus':
        open_set_data = load_imagenet21k(f'{dataset_dir}/imagenet21k_select2/')
    
    n_train = len(close_set_data['train']['labels'])
    indices = nrs.permutation(n_train)

    for lb_ID in [0.01, 0.02, 0.03, 0.05, 0.1, 0.15, 0.2]:

        n_lb = int(n_train * lb_ID)
        n_ub_ID = n_train-n_lb
        n_OOD = len(open_set_data)

        lb_idxs = indices[:n_lb]
        ub_idxs = indices[n_lb:n_lb+n_ub_ID]
        lb_train_imgs = close_set_data['train']['images'][lb_idxs]
        lb_train_labels = close_set_data['train']['labels'][lb_idxs]
        ub_train_imgs = np.concatenate((close_set_data['train']['images'][ub_idxs], open_set_data[:n_OOD]))
        open_set_labels = np.zeros((len(open_set_data), lb_train_labels.shape[1]))
        ub_train_labels = np.concatenate((close_set_data['train']['labels'][ub_idxs], open_set_labels[:n_OOD]))

        idx = nrs.permutation(len(ub_train_labels))
        ub_train_imgs = ub_train_imgs[idx]
        ub_train_labels = ub_train_labels[idx]

        print('generate data success!')
        print(f'{n_lb}:{n_ub_ID}:{n_OOD}')
        print(f'len(labeled data) {len(lb_idxs)}')
        print(f'len(unlabeled data) {len(ub_train_imgs)}')
        print(f'len(unlabeled close data) {len(ub_idxs)}')
        print(f"len(unlabeled open data) {n_OOD}")
        print(f"len(test data) {len(close_set_data['val']['labels'])}")
    
        if save:
            path = f'./data/{dataset_name}-imagenet/{seed}'
            os.makedirs(path, exist_ok=True)
            np.save(f'{path}/lb_image_{lb_ID}.npy', lb_train_imgs)
            np.save(f'{path}/lb_label_{lb_ID}.npy', lb_train_labels)
            np.save(f'{path}/ub_image_{lb_ID}.npy', ub_train_imgs)
            np.save(f'{path}/ub_label_{lb_ID}.npy', ub_train_labels)
            np.save(f'{path}/test_image.npy', close_set_data['val']['images'])
            np.save(f'{path}/test_label.npy', close_set_data['val']['labels'])


# use imagenet21k as open set and set ID num 1,2,3,4k
def gen_data_with_fixed_ID(dataset_dir, dataset_name, ID_d_OOD, seed, save=True):
    
    nrs = np.random.RandomState(seed)
    close_set_data = load_data(os.path.join(dataset_dir, dataset_name))
    if dataset_name == 'voc':
        close_set_data['train']['images'] = np.char.add(
            f'{dataset_dir}/voc/VOCdevkit/VOC2012/JPEGImages/', close_set_data['train']['images'])
        close_set_data['val']['images'] = np.char.add(
            f'{dataset_dir}/voc/VOCdevkit/VOC2012/JPEGImages/', close_set_data['val']['images'])
    if dataset_name == 'coco':
        close_set_data['train']['images'] = np.char.add(
            f'{dataset_dir}/coco/', close_set_data['train']['images'])
        close_set_data['val']['images'] = np.char.add(
            f'{dataset_dir}/coco/', close_set_data['val']['images'])
    if dataset_name == 'nus':
        close_set_data['train']['images'] = np.char.add(
            f'{dataset_dir}/nus/Flickr/', close_set_data['train']['images'])
        close_set_data['val']['images'] = np.char.add(
            f'{dataset_dir}/nus/Flickr/', close_set_data['val']['images'])
    
    if dataset_name == 'voc' or dataset_name == 'coco':
        open_set_data = load_imagenet21k(f'{dataset_dir}/imagenet21k_select1/')
    if dataset_name == 'nus':
        open_set_data = load_imagenet21k(f'{dataset_dir}/imagenet21k_select2/')
    
    n_train = len(close_set_data['train']['labels'])
    indices = nrs.permutation(n_train)

    if dataset_name=='nus':
        ID_num = [2000, 4000, 6000, 8000]
    elif dataset_name=='coco':
        ID_num = [1000, 2000, 3000, 4000]

    for lb_ID in ID_num:

        n_lb = lb_ID
        n_OOD = len(open_set_data)
        n_ub_ID = int(n_OOD * ID_d_OOD)
        if ID_d_OOD == 4.5:
            n_ub_ID = len(indices)

        lb_idxs = indices[:n_lb]
        ub_idxs = indices[n_lb:n_lb+n_ub_ID]
        lb_train_imgs = close_set_data['train']['images'][lb_idxs]
        lb_train_labels = close_set_data['train']['labels'][lb_idxs]
        ub_train_imgs = np.concatenate((close_set_data['train']['images'][ub_idxs], open_set_data[:n_OOD]))
        open_set_labels = np.zeros((len(open_set_data), lb_train_labels.shape[1]))
        ub_train_labels = np.concatenate((close_set_data['train']['labels'][ub_idxs], open_set_labels[:n_OOD]))

        idx = nrs.permutation(len(ub_train_labels))
        ub_train_imgs = ub_train_imgs[idx]
        ub_train_labels = ub_train_labels[idx]

        print('generate data success!')
        print(f'{n_lb}:{n_ub_ID}:{n_OOD}')
        print(f'len(labeled data) {len(lb_idxs)}')
        print(f'len(unlabeled data) {len(ub_train_imgs)}')
        print(f'len(unlabeled close data) {len(ub_idxs)}')
        print(f"len(unlabeled open data) {n_OOD}")
        print(f"len(test data) {len(close_set_data['val']['labels'])}")
    
        if save:
            path = f'./data/{dataset_name}-imagenet/{seed}'
            os.makedirs(path, exist_ok=True)
            np.save(f'{path}/lb_image_{lb_ID}_{ID_d_OOD}.npy', lb_train_imgs)
            np.save(f'{path}/lb_label_{lb_ID}_{ID_d_OOD}.npy', lb_train_labels)
            np.save(f'{path}/ub_image_{lb_ID}_{ID_d_OOD}.npy', ub_train_imgs)
            np.save(f'{path}/ub_label_{lb_ID}_{ID_d_OOD}.npy', ub_train_labels)
            np.save(f'{path}/test_image.npy', close_set_data['val']['images'])
            np.save(f'{path}/test_label.npy', close_set_data['val']['labels'])


# gen_data_without_openset('./dataset', 'coco', 4.5, 1, seed=1, save=True)
gen_data_with_in21k('./dataset', 'coco', seed=1, save=True)
# gen_data_with_fixed_ID('./dataset', 'nus', 4, seed=1, save=True)
