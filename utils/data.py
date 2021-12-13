import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from scipy.spatial.distance import squareform,pdist
from scipy.stats import multivariate_normal

def gen2Dgauss(x_mu=.0, y_mu=.0, xy_sigma=.1, n=20):
    """
    generates two-dimensional gaussian blob
    """
    xx,yy = np.meshgrid(np.linspace(0, 1, n),np.linspace(0, 1, n))
    gausspdf = multivariate_normal([x_mu,y_mu],[[xy_sigma,0],[0,xy_sigma]])
    x_in = np.empty(xx.shape + (2,))
    x_in[:, :, 0] = xx; x_in[:, :, 1] = yy
    return gausspdf.pdf(x_in)


def mk_block(context, do_shuffle):
    """
    generates block of experiment

    Input:
      - context  : 'task_a' or 'task_b'
      - do_shuffle: True or False, shuffles  values
    """
    resolution = 5
    n_units = resolution**2
    l, b = np.meshgrid(np.linspace(0.2, .8, 5),np.linspace(0.2, .8, 5))
    b = b.flatten()
    l = l.flatten()
    r_n, r_s = np.meshgrid(np.linspace(-2, 2, 5),np.linspace(-2, 2, 5))
    r_s = r_s.flatten()
    r_n = r_n.flatten()
    val_l, val_b = np.meshgrid(np.linspace(1, 5, 5),np.linspace(1, 5, 5))
    val_b = val_b.flatten()
    val_l = val_l.flatten()

    
    ii_sub = 1
    blobs = np.empty((25,n_units))
    for ii in range(0,25):
        blob = gen2Dgauss(x_mu=b[ii], y_mu=l[ii],xy_sigma=0.08,n=resolution)
        blob = blob/ np.max(blob)
        ii_sub += 1
        blobs[ii,:] = blob.flatten()
    x1 = blobs
    if context == 'task_a':        
        reward = r_n
    elif context == 'task_b':        
        reward = r_s

    feature_vals = np.vstack((val_b,val_l)).T
    if do_shuffle:
        ii_shuff = np.random.permutation(25)
        x1 = x1[ii_shuff,:]
        feature_vals = feature_vals[ii_shuff,:]
        reward = reward[ii_shuff]
    return x1, reward, feature_vals


def mk_block_humanchoices(context, do_shuffle, labels_a,labels_b,c_scaling=1):
    """
    generates block of experiment

    Input:
      - task  : 'task_a' or 'task_b'
      - do_shuffle: True or False, shuffles  values
    """
    resolution = 5
    n_units = resolution**2
    l, b = np.meshgrid(np.linspace(0.2, .8, 5),np.linspace(0.2, .8, 5))
    b = b.flatten()
    l = l.flatten()
    r_n = labels_a 
    r_s = labels_b
    val_l, val_b = np.meshgrid(np.linspace(1, 5, 5),np.linspace(1, 5, 5))
    val_b = val_b.flatten()
    val_l = val_l.flatten()

    #plt.figure()
    ii_sub = 1
    blobs = np.empty((25,n_units))
    for ii in range(0,25):
        blob = gen2Dgauss(x_mu=b[ii], y_mu=l[ii],xy_sigma=0.08,n=resolution)
        blob = blob/ np.max(blob)
        ii_sub += 1
        blobs[ii,:] = blob.flatten()
    x = blobs
    if context == 'task_a':
        x1 = np.append(blobs,c_scaling * np.ones((blobs.shape[0],1)),axis=1)
        x1 = np.append(x1,np.zeros((blobs.shape[0],1)),axis=1)
        reward = r_n
    elif context == 'task_b':
        x1 = np.append(blobs,np.zeros((blobs.shape[0],1)),axis=1)
        x1 = np.append(x1,c_scaling * np.ones((blobs.shape[0],1)),axis=1)
        reward = r_s

    feature_vals = np.vstack((val_b,val_l)).T
    if do_shuffle:
        ii_shuff = np.random.permutation(25)
        x1 = x1[ii_shuff,:]
        feature_vals = feature_vals[ii_shuff,:]
        reward = reward[ii_shuff]
    return x1, reward, feature_vals


def mk_block_wctx(context, do_shuffle, c_scaling=1):
    """
    generates block of experiment

    Input:
      - task  : 'task_a' or 'task_b'
      - do_shuffle: True or False, shuffles  values
    """
    resolution = 5
    n_units = resolution**2
    l, b = np.meshgrid(np.linspace(0.2, .8, 5),np.linspace(0.2, .8, 5))
    b = b.flatten()
    l = l.flatten()
    r_s, r_n = np.meshgrid(np.linspace(-2, 2, 5),np.linspace(-2, 2, 5))
    r_s = r_s.flatten()
    r_n = r_n.flatten()
    val_l, val_b = np.meshgrid(np.linspace(1, 5, 5),np.linspace(1, 5, 5))
    val_b = val_b.flatten()
    val_l = val_l.flatten()

    #plt.figure()
    ii_sub = 1
    blobs = np.empty((25,n_units))
    for ii in range(0,25):
        blob = gen2Dgauss(x_mu=b[ii], y_mu=l[ii],xy_sigma=0.08,n=resolution)
        blob = blob/ np.max(blob)
        ii_sub += 1
        blobs[ii,:] = blob.flatten()
    x = blobs
    if context == 'task_a':
        x1 = np.append(blobs,c_scaling * np.ones((blobs.shape[0],1)),axis=1)
        x1 = np.append(x1,np.zeros((blobs.shape[0],1)),axis=1)
        reward = r_n
    elif context == 'task_b':
        x1 = np.append(blobs,np.zeros((blobs.shape[0],1)),axis=1)
        x1 = np.append(x1,c_scaling * np.ones((blobs.shape[0],1)),axis=1)
        reward = r_s

    feature_vals = np.vstack((val_b,val_l)).T
    if do_shuffle:
        ii_shuff = np.random.permutation(25)
        x1 = x1[ii_shuff,:]
        feature_vals = feature_vals[ii_shuff,:]
        reward = reward[ii_shuff]
    return x1, reward, feature_vals




def make_dataset(args):
    """
    makes dataset for experiment
    """
    
    random_state = np.random.randint(999)

    x_task_a,y_task_a,f_task_a = mk_block_wctx('task_a',0,args.ctx_scaling)
    y_task_a = y_task_a[:,np.newaxis]
    l_task_a = (y_task_a>0).astype('int')

    x_task_b,y_task_b,f_task_b = mk_block_wctx('task_b',0,args.ctx_scaling)
    y_task_b = y_task_b[:,np.newaxis]
    l_task_b = (y_task_b>0).astype('int')

    if args.ctx_weights==True:
        x_task_a[:,:25] /= np.linalg.norm(x_task_a[:,:25])
        x_task_a[:,25:] /= np.linalg.norm(x_task_a[:,25:])
        x_task_b[:,:25] /= np.linalg.norm(x_task_b[:,:25])
        x_task_b[:,25:] /= np.linalg.norm(x_task_b[:,25:])
        
    x_in = np.concatenate((x_task_a,x_task_b),axis=0)
    y_rew = np.concatenate((y_task_a,y_task_b), axis=0)
    y_true = np.concatenate((l_task_a,l_task_b), axis=0)

    # define datasets (duplicates for shuffling)
    data = {}
    data['x_task_a'] = x_task_a
    data['y_task_a'] = y_task_a
    data['l_task_a'] = l_task_a

    data['x_task_b'] = x_task_b
    data['y_task_b'] = y_task_b
    data['l_task_b'] = l_task_b

    data['x_all'] = x_in
    data['y_all'] = y_rew
    data['l_all'] = y_true

    if args.training_schedule == 'interleaved':
        data['x_train'] = np.vstack(tuple([shuffle(data['x_all'],random_state = i+random_state) for i in range(args.n_episodes)]))
        data['y_train'] = np.vstack(tuple([shuffle(data['y_all'],random_state = i+random_state) for i in range(args.n_episodes)]))
        data['l_train'] = np.vstack(tuple([shuffle(data['l_all'],random_state = i+random_state) for i in range(args.n_episodes)]))
    elif args.training_schedule == 'blocked':
        data['x_train'] = np.vstack((
            np.vstack(tuple([shuffle(data['x_task_a'],random_state = i+random_state) for i in range(args.n_episodes)])),
            np.vstack(tuple([shuffle(data['x_task_b'],random_state = i+random_state) for i in range(args.n_episodes)]))))
        data['y_train'] = np.vstack((
            np.vstack(tuple([shuffle(data['y_task_a'],random_state = i+random_state) for i in range(args.n_episodes)])),
            np.vstack(tuple([shuffle(data['y_task_b'],random_state = i+random_state) for i in range(args.n_episodes)]))))
        data['l_train'] = np.vstack((
            np.vstack(tuple([shuffle(data['l_task_a'],random_state = i+random_state) for i in range(args.n_episodes)])),
            np.vstack(tuple([shuffle(data['l_task_b'],random_state = i+random_state) for i in range(args.n_episodes)]))))    
    else:
        print('Unknown training schedule parameter')
        

    if args.centering == True:
        sc = StandardScaler(with_std=False)
        data['x_train'] = sc.fit_transform(data['x_train'])
        data['x_task_a'] = sc.transform(data['x_task_a'])
        data['x_task_b'] = sc.transform(data['x_task_b'])
        x_in = StandardScaler(with_std=False).fit_transform(x_in)  
    
    if args.ctx_avg:
        if args.ctx_avg_type=='sma':             
            data['x_train'][:,-2] = pd.Series(data['x_train'][:,-2]).rolling(window=args.ctx_avg_window, min_periods=1).mean()
            data['x_train'][:,-1] = pd.Series(data['x_train'][:,-1]).rolling(window=args.ctx_avg_window, min_periods=1).mean()
            data['x_task_a'][:,-2] = pd.Series(data['x_task_a'][:,-2]).rolling(window=args.ctx_avg_window, min_periods=1).mean()
            data['x_task_a'][:,-1] = pd.Series(data['x_task_a'][:,-1]).rolling(window=args.ctx_avg_window, min_periods=1).mean()
            data['x_task_b'][:,-2] = pd.Series(data['x_trask_b'][:,-2]).rolling(window=args.ctx_avg_window, min_periods=1).mean()
            data['x_task_b'][:,-1] = pd.Series(data['x_trask_b'][:,-1]).rolling(window=args.ctx_avg_window, min_periods=1).mean()
        elif args.ctx_avg_type=='ema':
            data['x_train'][:,-2] = pd.Series(data['x_train'][:,-2]).ewm(alpha=args.ctx_avg_alpha, adjust=False, min_periods=1).mean()
            data['x_train'][:,-1] = pd.Series(data['x_train'][:,-1]).ewm(alpha=args.ctx_avg_alpha, adjust=False, min_periods=1).mean()
            data['x_task_a'][:,-2] = pd.Series(data['x_task_a'][:,-2]).ewm(alpha=args.ctx_avg_alpha, adjust=False, min_periods=1).mean()
            data['x_task_a'][:,-1] = pd.Series(data['x_task_a'][:,-1]).ewm(alpha=args.ctx_avg_alpha, adjust=False, min_periods=1).mean()
            data['x_task_b'][:,-2] = pd.Series(data['x_task_b'][:,-2]).ewm(alpha=args.ctx_avg_alpha, adjust=False, min_periods=1).mean()
            data['x_task_b'][:,-1] = pd.Series(data['x_task_b'][:,-1]).ewm(alpha=args.ctx_avg_alpha, adjust=False, min_periods=1).mean()

    return data






def make_dataset_fromchoices(args,choices_a,choices_b):
    """
    makes dataset for experiment, human choice patterns as labels
    """
    choices_a = choices_a.flatten()
    choices_b = choices_b.flatten()

    random_state = np.random.randint(999)

    x_task_a,y_task_a,f_task_a = mk_block_humanchoices('task_a',0,choices_a,choices_b,args.ctx_scaling)
    y_task_a = y_task_a[:,np.newaxis]
    l_task_a = (y_task_a>0).astype('int')

    x_task_b,y_task_b,f_task_b = mk_block_humanchoices('task_b',0,choices_a,choices_b,args.ctx_scaling)
    y_task_b = y_task_b[:,np.newaxis]
    l_task_b = (y_task_b>0).astype('int')

    x_in = np.concatenate((x_task_a,x_task_b),axis=0)
    y_rew = np.concatenate((y_task_a,y_task_b), axis=0)
    y_true = np.concatenate((l_task_a,l_task_b), axis=0)

    # define datasets (duplicates for shuffling)
    data = {}
    data['x_task_a'] = x_task_a
    data['y_task_a'] = y_task_a
    data['l_task_a'] = l_task_a

    data['x_task_b'] = x_task_b
    data['y_task_b'] = y_task_b
    data['l_task_b'] = l_task_b

    data['x_all'] = x_in
    data['y_all'] = y_rew
    data['l_all'] = y_true

    if args.training_schedule == 'interleaved':
        data['x_train'] = np.vstack(tuple([shuffle(data['x_all'],random_state = i+random_state) for i in range(args.n_episodes)]))
        data['y_train'] = np.vstack(tuple([shuffle(data['y_all'],random_state = i+random_state) for i in range(args.n_episodes)]))
        data['l_train'] = np.vstack(tuple([shuffle(data['l_all'],random_state = i+random_state) for i in range(args.n_episodes)]))
    elif args.training_schedule == 'blocked':
        data['x_train'] = np.vstack((
            np.vstack(tuple([shuffle(data['x_task_a'],random_state = i+random_state) for i in range(args.n_episodes)])),
            np.vstack(tuple([shuffle(data['x_task_b'],random_state = i+random_state) for i in range(args.n_episodes)]))))
        data['y_train'] = np.vstack((
            np.vstack(tuple([shuffle(data['y_task_a'],random_state = i+random_state) for i in range(args.n_episodes)])),
            np.vstack(tuple([shuffle(data['y_task_b'],random_state = i+random_state) for i in range(args.n_episodes)]))))
        data['l_train'] = np.vstack((
            np.vstack(tuple([shuffle(data['l_task_a'],random_state = i+random_state) for i in range(args.n_episodes)])),
            np.vstack(tuple([shuffle(data['l_task_b'],random_state = i+random_state) for i in range(args.n_episodes)]))))    
    else:
        print('Unknown training schedule parameter')
        

    if args.ctx_avg:    
        data['x_train'][:,-2] = pd.Series(data['x_train'][:,-2]).rolling(window=args.ctx_avg_window, min_periods=1).mean()
        data['x_train'][:,-1] = pd.Series(data['x_train'][:,-1]).rolling(window=args.ctx_avg_window, min_periods=1).mean()

    if args.centering == True:
        sc = StandardScaler(with_std=False)
        data['x_train'] = sc.fit_transform(data['x_train'])
        data['x_task_a'] = sc.transform(data['x_task_a'])
        data['x_task_b'] = sc.transform(data['x_task_b'])
        x_in = StandardScaler(with_std=False).fit_transform(x_in)  
    return data