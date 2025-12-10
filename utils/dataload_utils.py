import pandas as pd
import numpy as np
import json
import os
import yaml
from sympy.parsing.sympy_parser import parse_expr
from sympy import symbols

def load_posteriordb_data(data_name, return_origXY=False):
    # https://github.com/stan-dev/posteriordb.git
    path = f'/home/mok/module/icml25/posteriordb/posterior_database/data/data/{data_name}.json'
    
    with open(path) as f:
        data = json.load(f)
    
    if 'Y' in data:
        origX = data['x']
        origY = data['Y']
        origX = np.expand_dims(origX, axis=1)
        origY = np.expand_dims(origY, axis=1)
        origY= (origY-np.min(origY))/(np.max(origY-np.min(origY)))
        num_val = 3
        num_test = 4
        
    elif 'y' in data:
        origY = data['y']
        origX = np.linspace(0, len(origY), len(origY))
        origX = np.expand_dims(origX, axis=1)
        origY = np.expand_dims(origY, axis=1)
        num_val = int(len(origY)*0.1)
        num_test = int(len(origY)*0.1)
        
    elif 'n' in data:
        origY = data['n']
        origX = np.linspace(0, len(origY), len(origY))
        origX = np.expand_dims(origX, axis=1)
        origY = np.expand_dims(origY, axis=1)
        num_val = int(len(origY)*0.1)
        num_test = int(len(origY)*0.1)
        
    if num_val==0:
        num_val = 1
    if num_test==0:
        num_test = 1
        
        
    trainX = origX[:-1*(num_val+num_test)]
    trainY = origY[:-1*(num_val+num_test)]
    valX = origX[-1*(num_val+num_test):-1*num_test]
    valY = origY[-1*(num_val+num_test):-1*num_test]
    testX = origX[-1*num_test:]
    testY = origY[-1*num_test:]
    
    X = trainX
    Y = trainY
    
    
    data  = {'X': [round(x[0], 2) for x in X], 'Y': [round(y[0], 2) for y in Y]}
    range_ = np.max(Y)-np.min(Y)
        
    if not return_origXY:  
        return X, Y, valX, valY, testX, testY, data, range_
    
    else:
        return X, Y, valX, valY, testX, testY, data, range_, origX, origY
    

def load_gp_data(data_name, return_origXY=False, noise=0):
    path = f'/home/mok/module/icml25/gpss-research/data/tsdlr_9010_csv/mok'
    train_df = pd.read_csv(f'{path}/{data_name}-train.csv')
    test_df = pd.read_csv(f'{path}/{data_name}-test.csv')
    
    all_ = np.array(train_df)
    X = np.expand_dims(all_[:, 0], axis=1)
    Y = np.expand_dims(all_[:, 1], axis=1)
    
    testall_ = np.array(test_df)
    if noise!=0:
        Y += np.random.randn(Y.shape[0], 1)*noise*(np.max(Y)-np.min(Y))
    
    testX = np.expand_dims(testall_[:, 0], axis=1)
    testY = np.expand_dims(testall_[:, 1], axis=1)
    
    origX = np.concatenate([X, testX])
    origY = np.concatenate([Y, testY])
    
    ###### normalize
    normed_origY= (origY-np.min(origY))/(np.max(origY-np.min(origY)))
    normed_Y = normed_origY[:Y.shape[0]]
    normed_testY = normed_origY[Y.shape[0]:]
    
    Y = normed_Y
    testY = normed_testY
    origY = normed_origY
    
    # normed_origX= (origX-np.min(origX))/(np.max(origX-np.min(origX)))
    # normed_X = normed_origX[:X.shape[0]]
    # normed_testX = normed_origX[X.shape[0]:]
    
    # X = normed_X
    # testX = normed_testX
    # origX = normed_origX
    
    ###### split train and val
    split_point = int(0.9 * len(X))  # 배열 길이의 70% 위치 계산
    trainX = X[:split_point]
    valX = X[split_point:]
    X = trainX
    
    trainY = Y[:split_point]
    valY = Y[split_point:]
    Y = trainY
    
    data  = {'X': [round(x[0], 2) for x in X], 'Y': [round(y[0], 2) for y in Y]}
    range_ = np.max(Y)-np.min(Y)
    
    if not return_origXY:  
        return X, Y, valX, valY, testX, testY, data, range_
    
    else:
        return X, Y, valX, valY, testX, testY, data, range_, origX, origY

def load_m3_data(data_name, return_origXY=False):
    # path = f'/home/mok/module/abcd/ASMD/ablation/m3_data/M3Forecast.xls'
    # train_df = pd.read_csv(f'{path}')
    path = f'/home/mok/module/icml25/ASMD/ablation/m3_and_tmp_others/M3C.xls'
    df = pd.read_excel(f'{path}', sheet_name='M3Month')
    df = df[df['Series']==data_name].iloc[:, 6:].values[0]

    all_ = np.array(df)

    # X = np.expand_dims(all_[:, 0], axis=1)
    X = np.expand_dims(np.arange(0, len(all_)), axis=1)
    Y = np.expand_dims(all_, axis=1)
    
    # origX = np.concatenate([X, testX])
    # origY = np.concatenate([Y, testY])
    origX = X
    origY = Y
    
    ###### normalize
    normed_origY= (origY-np.min(origY))/(np.max(origY-np.min(origY)))
    normed_Y = normed_origY[:-18]
    normed_testY = normed_origY[-18:]
    
    Y = normed_Y
    testY = normed_testY
    origY = normed_origY
    
    trainX = X[:-18]
    testX = X[-18:]
    ###### split train and val
    split_point = int(0.9 * len(trainX))  # 배열 길이의 70% 위치 계산
    valX = trainX[split_point:]
    trainX = trainX[:split_point]

    trainY = Y[:split_point]
    valY = Y[split_point:]
    
    X = trainX
    Y = trainY
    
    data  = {'X': [round(x[0], 2) for x in X], 'Y': [round(y[0], 2) for y in Y]}
    range_ = np.max(Y)-np.min(Y)
    
    if not return_origXY:  
        return X, Y, valX, valY, testX, testY, data, range_
    
    else:
        return X, Y, valX, valY, testX, testY, data, range_, origX, origY
    
def load_points(file_path: str) -> np.ndarray:
    if file_path.endswith(".npy"):
        points = np.load(file_path)
    elif file_path.endswith(".txt"):
        points = np.loadtxt(file_path)
    elif file_path.endswith(".csv"):
        points = pd.read_csv(file_path).values
    elif file_path.endswith(".tsv"):
        points = pd.read_csv(file_path, sep="\t").values
    else:
        raise ValueError("Invalid file format. (only .npy, .txt, .csv, and .tsv are supported)")
    return points

def normalize_points(points: np.ndarray, method: str = "minmax", percentile: int = None) -> np.ndarray:
    if method == "percentile" and percentile is None:
        raise ValueError("Percentile normalization requires a percentile value.")
    
    ys = np.array([point[-1] for point in points])
    if method == "minmax":
        points = np.array([np.concatenate([point[:-1], [(y - ys.min()) / (ys.max() - ys.min())]]) for point, y in zip(points, ys)])
    elif method == "zscore":
        points = np.array([np.concatenate([point[:-1], [(y - ys.mean()) / ys.std()]]) for point, y in zip(points, ys)])
    elif method == "percentile":
        points = np.array([np.concatenate([point[:-1], [y /np.percentile(ys, percentile)]]) for point, y in zip(points, ys)])
    else:
        raise ValueError("Invalid normalization method.")

    points = np.round(points, 4)
    return points

def load_mgp_data(data_name, return_origXY=False, noise=0):
    # path = f'/home/mok/module/icml25/ASMD/multivariate-time-series-data/ICVLResnet101FeaturesandLabels'
    # train_df = pd.read_csv(f'{path}/Trn.csv')
    # test_df = pd.read_csv(f'{path}/Tst.csv')
    path = f'/home/mok/module/icml25/ASMD/multivariate-time-series-data/{data_name}/{data_name}.txt'
    df = np.loadtxt(path, delimiter=',')
    train_df = df[:int(df.shape[0]*2/3), :]
    test_df = df[int(df.shape[0]*2/3):, :]
        
    all_ = np.array(train_df)
    # X = np.expand_dims(all_[:, :7], axis=1)
    X = all_[:, :8]
    Y = np.expand_dims(all_[:, -1], axis=1)
    
    testall_ = np.array(test_df)
    if noise!=0:
        Y += np.random.randn(Y.shape[0], 1)*noise*(np.max(Y)-np.min(Y))
    
    # testX = np.expand_dims(testall_[:, :7], axis=1)
    testX = testall_[:, :8]
    testY = np.expand_dims(testall_[:, -1], axis=1)
    
    origX = np.concatenate([X, testX])
    origY = np.concatenate([Y, testY])
    
    ###### normalize
    normed_origY= (origY-np.min(origY))/(np.max(origY-np.min(origY)))
    normed_Y = normed_origY[:Y.shape[0]]
    normed_testY = normed_origY[Y.shape[0]:]
    
    Y = normed_Y
    testY = normed_testY
    origY = normed_origY
    
    normed_origX= (origX-np.min(origX))/(np.max(origX-np.min(origX)))
    normed_X = normed_origX[:X.shape[0]]
    normed_testX = normed_origX[X.shape[0]:]
    
    X = normed_X
    testX = normed_testX
    origX = normed_origX
    
    ###### split train and val
    split_point = int(0.9 * len(X))  # 배열 길이의 70% 위치 계산
    trainX = X[:split_point]
    valX = X[split_point:]
    X = trainX
    
    trainY = Y[:split_point]
    valY = Y[split_point:]
    Y = trainY
    
    data  = {'X': [round(x[0], 2) for x in X], 'Y': [round(y[0], 2) for y in Y]}
    range_ = np.max(Y)-np.min(Y)
    
    if not return_origXY:  
        return X, Y, valX, valY, testX, testY, data, range_
    
    else:
        return X, Y, valX, valY, testX, testY, data, range_, origX, origY

def load_sr_data(data_folder, return_origXY=True):
    train_points_file = os.path.join(data_folder, 'train_points.npy')
    train_points = load_points(train_points_file)
    train_points = normalize_points(train_points)
    num_train_points = len(train_points)
    min_train_points = np.min(train_points)
    max_train_points = np.max(train_points)
    
    test_points_file = os.path.join(data_folder, 'test_points.npy')
    test_points = load_points(test_points_file)
    test_points = normalize_points(test_points)
    num_test_points = len(test_points)
    min_test_points = np.min(test_points)
    max_test_points = np.max(test_points)
    
    X = train_points[:, 0]
    Y = train_points[:, 1]
    
    split_point = int(0.9 * len(X))
    trainX = X[:split_point]
    valX = X[split_point:]
    trainY = Y[:split_point]
    valY = Y[split_point:]
    
    testX = test_points[:, 0]
    testY = test_points[:, 1]
    
    data  = {'X': [round(x, 2) for x in trainX], 'Y': [round(y, 2) for y in trainY]}
    range_ = np.max(trainY)-np.min(trainY)
    
    if return_origXY:
        if not train_points.shape[-1]==2:
            print(data_folder, 'is multi dim')
            return None, None, None
        xmin = np.min([a for a,b in train_points])
        xmax = np.max([a for a,b in train_points])
        xrange = xmax-xmin
        testx1 = np.arange(xmin-xrange*0.5, xmin, 0.05)
        testx2 = np.arange(xmax, xmax+xrange*0.5, 0.05)
        
        dataname_num = data_folder.split('/')[-1]
        c = list(filter(str.isalpha, dataname_num))
        c =''.join(c)
        
        gt_function_path = f'/home/mok/module/icml25/ASMD/ablation/ablate_models/sr/In-Context-Symbolic-Regression/conf/experiment/function/{c}/{dataname_num}.yaml'
        with open(gt_function_path) as f: 
            tmp=yaml.load(f, Loader=yaml.FullLoader)
                        
        gt_function = tmp['test_function']
        gt_function = parse_expr(gt_function.replace('^', '**'))
        
        # testx_ood = np.concatenate([testx1, testx2], axis=0)
        testx_ood = np.arange(xmax, xmax+xrange*0.1, 0.01)
        
        x = symbols('x')
        testy_ood = []
        for gtx in testx_ood:
            gt_pred = gt_function.evalf(subs={x: gtx})
            testy_ood.append(gt_pred)
            
        try: testy_ood = np.asarray(testy_ood).astype(float)
        except: 
            print('not working')
            testy_ood = None
        
        if testy_ood is not None:
            test_points_ood = [[a, b] for a, b in zip(testx_ood, testy_ood)]
            test_points_ood = np.asarray(test_points_ood)
            test_oodX = test_points_ood[:, 0]
            test_oodY = test_points_ood[:, 1]
        
        else: test_points_ood = None
        
        origX = np.concatenate([X, test_oodX])
        origY = np.concatenate([Y, test_oodY])
        
    trainX = np.expand_dims(trainX, axis=1)
    trainY = np.expand_dims(trainY, axis=1)
    valX = np.expand_dims(valX, axis=1)
    valY = np.expand_dims(valY, axis=1)
    testX = np.expand_dims(testX, axis=1)
    testY = np.expand_dims(testY, axis=1)
    origX = np.expand_dims(origX, axis=1)
    origY = np.expand_dims(origY, axis=1)
    
    # return train_points, num_train_points, min_train_points, max_train_points, test_points, num_test_points, min_test_points, max_test_points
    if not return_origXY:  
        return trainX, trainY, valX, valY, testX, testY, data, range_
    
    else:
        return trainX, trainY, valX, valY, testX, testY, data, range_, origX, origY

def mse(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def nmse(y_true, y_pred):
    return np.sum((np.array(y_true) - np.array(y_pred)) ** 2)/np.sum(np.array(y_true)**2)

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))

def r2_score(y_true, y_pred):
    """
    R² (결정계수)를 계산하는 함수.
    
    Parameters:
    - y_true: 실제 값 (리스트나 numpy 배열)
    - y_pred: 예측 값 (리스트나 numpy 배열)
    
    Returns:
    - R² 값 (float)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    return 1 - (ss_res / ss_tot)

