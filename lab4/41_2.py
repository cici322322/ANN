from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet
import math

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    ''' restricted boltzmann machine '''
    
    print ("\nStarting a Restricted Boltzmann Machine..")

    #set minibatch size and dimension of hidden layer
    batch_size = 20
    ndim_hidden = 500

    #set number of epochs
    epochs = 1;

    # number of iterations to pass through the whole dataset
    n_iterations = math.floor(60000/batch_size)

    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden = ndim_hidden,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size = batch_size
    )

    reconstr_err_500 = []

    for i in range(0,epochs):
        rbm.cd1(visible_trainset=train_imgs, n_iterations=n_iterations)
        reconstr_err_500 += rbm.get_err_rec()

    #set minibatch size and dimension of hidden layer
    batch_size = 20
    ndim_hidden = 200


    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0]*image_size[1],
                                     ndim_hidden = ndim_hidden,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size = batch_size
    )

    reconstr_err_200 = []

    for i in range(0,epochs):
        rbm.cd1(visible_trainset=train_imgs, n_iterations=n_iterations)
        reconstr_err_200 += rbm.get_err_rec()


    x_axis = np.arange(len(reconstr_err_500))/n_iterations
    plt.title("Reconstruction error along epochs") 
    plt.xlabel("Epoch") 
    plt.ylabel("Reconstruction error") 
    plt.plot(x_axis,reconstr_err_500,x_axis,reconstr_err_200)
    plt.gca().legend(('500','200')) 
    plt.show()
    