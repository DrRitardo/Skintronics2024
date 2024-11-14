import numpy as np
import matplotlib.pyplot as plt

# This section is the uses the nearest centroid classification method to classify the alphabets

foldername = '2024_06_14/up' #'2023_06_08'
#filename_vec = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
#                'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
filename_vec = ['u', 'c', 'l', 'a']
num_alphabets = 4
test_ratio = 0.2

def load_shuffle():
    for idx in range(num_alphabets):
        new_X = np.loadtxt('data/' + foldername + '/' + filename_vec[idx] + '_stack.txt')
        new_Y = idx * np.ones(new_X.shape[1])
        if idx == 0:
            X_stack = new_X
            Y_stack = new_Y
        else:
            X_stack = np.append(X_stack, new_X, axis=1)
            Y_stack = np.append(Y_stack, new_Y)
    full_stack = np.append(Y_stack.reshape((1,-1)), X_stack, axis=0)
    full_stack = np.transpose(full_stack)
    np.random.shuffle(full_stack)
    shuffled_stack = np.transpose(full_stack)

    return shuffled_stack

def separate_train_test(shuffled_stack, test_ratio, test_choice):
    N_total_cases = shuffled_stack.shape[1]
    N_test = int(N_total_cases * test_ratio)
    test_stack = shuffled_stack[:, N_test * test_choice:N_test * (test_choice + 1)]
    if test_choice == 0:
        train_stack = shuffled_stack[:, N_test:N_total_cases]
    elif test_choice == int(1 / test_ratio - 1):
        train_stack = shuffled_stack[:, 0:N_test * test_choice]
    else:
        train_stack = np.append(shuffled_stack[:, 0:N_test * test_choice],
                                shuffled_stack[:, N_test * (test_choice + 1):N_total_cases], axis=1)
    return train_stack, test_stack

def get_mean_vecs(input_stack):
    mean_vecs = np.zeros((input_stack.shape[0]-1, num_alphabets))
    label_cnts = np.zeros(num_alphabets)
    print(input_stack.shape)
    print(input_stack.shape[1])
    for idx in range(input_stack.shape[1]):
        for j in range(num_alphabets):
            if j == input_stack[0, idx]:
                mean_vecs[:, j] += input_stack[1:input_stack.shape[0], idx]
                label_cnts[j] += 1
    for j in range(num_alphabets):
        mean_vecs[:, j] = mean_vecs[:, j]/label_cnts[j]
    return mean_vecs

def accumulate_one_shuffle(mean_stack, test_stack):
    classify_bin = np.zeros((num_alphabets, num_alphabets))
    cmp_vec = np.zeros(num_alphabets)
    for k in range(test_stack.shape[1]):
        for j in range(num_alphabets):
            vec1 = mean_stack[:, j]
            vec2 = test_stack[1:test_stack.shape[0], k]
            #norm_conv = np.convolve(vec1, np.flip(vec2))/np.sqrt(np.mean(vec1 ** 2) * np.mean(vec2 ** 2))
            #cmp_vec[j] = np.amax(norm_conv)
            norm_dot_product = vec1 * vec2 / np.sqrt(np.mean(vec1 ** 2) * np.mean(vec2 ** 2))
            cmp_vec[j] = np.sum(norm_dot_product)
        idx_cmp = np.argmax(cmp_vec)
        #print(cmp_vec)
        classify_bin[int(test_stack[0, k]), idx_cmp] += 1
    return classify_bin

shuffled_stack = load_shuffle()
total_classify_bin = np.zeros((num_alphabets, num_alphabets))
total_cnt_bin = np.zeros(num_alphabets)
for cycle in range(int(1/test_ratio)):
    train_stack, test_stack = separate_train_test(shuffled_stack, test_ratio, test_choice=cycle)
    mean_stack = get_mean_vecs(train_stack)
    classify_bin = accumulate_one_shuffle(mean_stack, test_stack)
    total_classify_bin += classify_bin
for idx in range(num_alphabets):
    total_classify_bin[idx, :] = total_classify_bin[idx, :] / np.sum(total_classify_bin[idx, :])



plt.imshow(total_classify_bin, vmax=1, vmin=0)
plt.colorbar()
plt.axis('off')
plt.show()