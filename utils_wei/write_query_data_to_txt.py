
# parameter data is from one query , and the last column is label
def write_to_txt(X, restore_path):
    
    with open(restore_path, 'w') as f:
        for i in range(X.shape[0]):
            line = ""
            line = line + str(int(X[i][-1].item())) + ' ' + 'qid:' + str(i) + ' '
            for j in range(X.shape[1]-1):
                line += ((str(j+1)) + ":" + str(X[i, j].item()) + " ")
            line = line + "\n"
            f.write(line)
