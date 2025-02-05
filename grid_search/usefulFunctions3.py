'''
hyperparamètres :
- batch_size : [2;100]
- num_epoch : [10;50]
- nz : [1;50]
- lr : [1e-4;1e-5]
- beta1 : [0;1]
'''

def getGrid3(batch_sizes, num_epochs) :
    return [[(batch_size, num_epoch) for batch_size in batch_sizes] for num_epoch in num_epochs]