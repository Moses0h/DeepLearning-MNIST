from fashion_mnist_dataset.utils import mnist_reader
import numpy as np
from numpy import linalg
import random
from matplotlib import pyplot as plt

NUM_SETS = 10
NUM_PC = 200
NUM_EPOCH = 100
BATCH_SIZE = 512
LEARNING_RATE = 0.0006 
THRESHOLD = -0.01 # Negative = lower threshold

def shuffle(X, Y):
    assert len(X) == len(Y)
    
    joined = list(zip(X, Y))
    random.shuffle(joined)
    return zip(*joined)


def one_hot_encode(num, max_num=10):    
    encoding = np.zeros((num.size, num.max() + 1))
    encoding[np.arange(num.size), num] = 1
    return encoding


def min_max_normalize(X):
    X = np.copy(X)
    return (X - X.min(axis = 0))/ (X.max(axis = 0) - X.min(axis = 0))


def cross_validation_procedure(x_folds, y_folds, PC):
    train_set_loss, val_set_loss, train_set_accuracy, val_set_accuracy = [], [], [], []
    # 4: for fold = 1 to k do
    for i in range(len(x_folds)):
        # 5: val_set ← folds[fold];
        X_val_set = x_folds[i]
        y_val_set = y_folds[i]

        # 6: train_set ← remaining folds;

        X_train_set = np.vstack(np.delete(x_folds, i, 0))
        y_train_set = np.vstack(np.delete(y_folds, i, 0))

        ''' 
        LOGISTIC REGRESSION
        
        y_train_set = np.hstack(np.delete(y_folds, i, 0))
        '''
        


        # 7: Project train_set and val_set onto top p train PC’s
        X_train_set = projection(X_train_set, PC)
        X_train_set = np.insert(X_train_set, 0, 1, axis = 1)

        X_val_set = projection(X_val_set, PC)
        X_val_set = np.insert(X_val_set, 0, 1, axis = 1)

        # 9.2: w ← 0
        best_accuracy = 0
        best_weights = None

        #logistic regression
        # w = np.zeros(len(X_train_set[0]))
        w = np.zeros([len(X_train_set[0]), len(y_train_set[0])])


        prev_loss = np.inf

        epoch_train_loss, epoch_val_loss, epoch_train_accuracy, epoch_val_accuracy = [], [], [], []

        # 8: for epoch = 1 to M do
        for t in range (NUM_EPOCH):
            # 9: train the model with train_set, and test the performance on val_set every epoch
            # 9.4: randomize the order of the indices into the training set   
            # 9.5

            '''
            LOGISTIC REGRESSION

            # current_loss = cross_entropy_cost_function(X_train_set, y_train_set, w)
            '''

            current_loss = softmax_loss(y_train_set, X_train_set, w)
            w = gradient_descent(X_train_set, y_train_set, w)
            train_set_prediction = softmax(np.matmul(X_train_set, w))
            val_set_prediction = softmax(np.matmul(X_val_set, w))

            train_accuracy = sum([1 if all(y_train_set[i] == np.round(train_set_prediction[i])) else 0 for i in range(len(train_set_prediction))])/len(train_set_prediction)
            val_accuracy = sum([1 if all(y_val_set[i] == np.round(val_set_prediction[i])) else 0 for i in range(len(val_set_prediction))])/len(val_set_prediction)
            
            #10: record train_set, val_set loss for plotting and accuracy on val_set for hyperparameter tuning
            epoch_train_loss.append(current_loss)

            val_loss = softmax_loss(y_val_set, X_val_set, w)
            #val_loss = cross_entropy_cost_function(X_val_set, y_val_set, w)
            epoch_val_loss.append(val_loss)
            epoch_train_accuracy.append(train_accuracy)
            epoch_val_accuracy.append(val_accuracy)            

            if val_accuracy > best_accuracy:
                best_weights = w
                best_accuracy = val_accuracy

            if (prev_loss - current_loss < THRESHOLD):
                print("Ending at epoch:", t , "|Accuracy|train:", train_accuracy, "|val:", val_accuracy)
                break

            prev_loss = current_loss
        print("--------------------------------------------------------------------------------------------------------")
        train_set_loss.append(epoch_train_loss)
        val_set_loss.append(epoch_val_loss)
        train_set_accuracy.append(epoch_train_accuracy)
        val_set_accuracy.append(epoch_val_accuracy)
        # print(best_accuracy)
    return best_weights, train_set_loss, val_set_loss, train_set_accuracy, val_set_accuracy

# X is the data sets (train), and k is the number of principal components we want returned
def PCA(X, k):
    M = np.mean(X)
    X_centered = X - M
    U, s, Vh = np.linalg.svd(X_centered, full_matrices=False)
    V = np.transpose(Vh)
    #return top k Principal Components
    return V[:, :k]

#A * Wk * Wtk

def projection(X, PC):
    return np.matmul(X, PC)

def reconstruction(X, PC):
    return np.matmul(projection(X, PC), np.transpose(PC))

def cross_entropy_cost_function(x, y, w):
    sum = 0
    for i in range (len(x)):
        yh = sigmoid(np.matmul(np.transpose(w), x[i]))
        sum += y[i] * np.log(yh) + (1 - y[i]) * np.log(1 - yh)
    return -sum/(len(x)) 
    
def cross_function_gradient(x, y, w, start, end):
    yh = softmax(np.matmul(x[start:end], w))
    return np.matmul(np.transpose(x[start:end]), (yh - y[start:end]))


def softmax(y):
    # Compute softmax values for each sets of scores in y.
    ps = np.empty(y.shape)
    for i in range(y.shape[0]):
        ps[i] = np.exp(y[i] - np.max(y[i]))
        ps[i] /= ps[i].sum()
    return ps


def softmax_loss(y, x, w):
    yh = softmax(np.matmul(x, w))
    sum = np.sum(np.log(y + 1e-6) * (yh))
    return -sum / (len(x) * len(y[0]))

def gradient_descent(x_train, y_train, w):  
    shuffled_x, shuffled_y = shuffle(x_train, y_train)
    # 5: for j = 1 to N, in steps of B do . Here, N is number of examples and B is the batch size
    for j in range (0, len(shuffled_x), BATCH_SIZE):
        start = j
        end = j + BATCH_SIZE
    
        if (end > len(shuffled_x)):
            end = len(shuffled_x)
        
        w -= LEARNING_RATE * cross_function_gradient(shuffled_x, shuffled_y, w, start, end)
    return w

def sigmoid(i):
    return 1/(1 + np.exp(-i))

def display_multiple_img(images, rows = 1, cols=1):
    figure, ax = plt.subplots(nrows=rows,ncols=cols )
    for ind,title in enumerate(images):
        ax.ravel()[ind].imshow(images[title])
        ax.ravel()[ind].set_title(title)
        ax.ravel()[ind].set_axis_off()
    plt.tight_layout()
    plt.show()

def ml_pipeline():
    X_train , y_train = mnist_reader.load_mnist ( "fashion_mnist_dataset/data/fashion", kind ='train')
    X_test , y_test = mnist_reader.load_mnist ("fashion_mnist_dataset/data/fashion", kind = 't10k')

    X_train = min_max_normalize(X_train)
    X_test = min_max_normalize(X_test)

    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)


    '''
    # LOGISTIC REGRESSION CODE

    filtered_train_indices = [True if i == 0 or i == 9 else False for i in y_train]
    filtered_test_indices = [True if i == 0 or i == 9 else False for i in y_test]

    X_train = X_train[filtered_train_indices]
    y_train = y_train[filtered_train_indices]
    y_train = np.array([1 if i == 0 else 0 for i in y_train])

    X_test = X_test[filtered_test_indices]
    y_test = y_test[filtered_test_indices]
    y_test = np.array([1 if i == 0 else 0 for i in y_test])

    SECOND DATA SET

    filtered_train_indices = [True if i == 2 or i == 4 else False for i in y_train]
    filtered_test_indices = [True if i == 2 or i == 4 else False for i in y_test]

    X_train = X_train[filtered_train_indices]
    y_train = y_train[filtered_train_indices]
    y_train = np.array([1 if i == 2 else 0 for i in y_train])

    X_test = X_test[filtered_test_indices]
    y_test = y_test[filtered_test_indices]
    y_test = np.array([1 if i == 2 else 0 for i in y_test])

    '''

    x_train_shuffled, y_train_shuffled = shuffle(X_train, y_train)

    #2: Perform PCA on the whole train
    PC = PCA(x_train_shuffled, NUM_PC)

    # 3: folds = k mutex split of training data;
    x_train_sets = np.array_split(x_train_shuffled, NUM_SETS)
    y_train_sets = np.array_split(y_train_shuffled, NUM_SETS)

    best_weights, train_loss, val_loss, train_accuracy, val_accuracy = cross_validation_procedure(x_train_sets, y_train_sets, PC)

    X_test = projection(X_test, PC)

    X_test = np.insert(X_test, 0, 1, axis = 1)

    test_set_prediction = softmax(np.matmul(X_test, best_weights))

    test_accuracy = sum([1 if np.argmax(test_set_prediction[i]) == np.argmax(y_test[i]) else 0 for i in range(y_test.shape[0])])/y_test.shape[0]
    print("Test Set Accuracy with Best Weight:", test_accuracy)


    best_weights = np.delete(best_weights, 0, axis = 0)
    best_weights_transposed = np.transpose(best_weights)
    
    # Show Weights
    for i in range(len(best_weights_transposed)):
        best_weights_reconstructed = projection(best_weights_transposed[i], np.transpose(PC))
        x = plt.matshow(best_weights_reconstructed.reshape(28, 28), cmap='Oranges')
        plt.colorbar(x)
        plt.show()

    max_len = max(len(row) for row in train_loss)
    train_transpose = [[row[col] for row in train_loss if len(row) > col] for col in range(max_len)]

    train_mean = [np.mean(np.array(i)) for i in train_transpose]
    train_std = [np.std(np.array(i)) for i in train_transpose]

    max_len = max(len(row) for row in val_loss)
    val_transpose = [[row[col] for row in val_loss if len(row) > col] for col in range(max_len)]

    val_mean = [np.mean(np.array(i)) for i in val_transpose]
    val_std = [np.std(np.array(i)) for i in val_transpose]


    plt.errorbar(range(len(train_transpose)), train_mean, train_std, color = 'blue', label='train loss', capsize=3)
    plt.errorbar(range(len(val_transpose)), val_mean, val_std, color = 'red', label='val loss', capsize=3)
    plt.xlabel('Epoch Number')
    plt.ylabel('Average Loss')
    plt.title('Average Loss Across Classes For Each Epoch')
    plt.legend()
    plt.show()



ml_pipeline()
