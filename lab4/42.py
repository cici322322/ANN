from util import *
from rbm import RestrictedBoltzmannMachine 
from dbn import DeepBeliefNet

if __name__ == "__main__":

    image_size = [28,28]
    train_imgs,train_lbls,test_imgs,test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)
    
    ''' deep- belief net '''

    print ("\nStarting a Deep Belief Net..")

    batch_size = 10
    
    dbn = DeepBeliefNet(sizes={"vis":image_size[0]*image_size[1], "hid":500, "pen":500, "top":2000, "lbl":10},
                        image_size=image_size,
                        n_labels=10,
                        batch_size=batch_size
    )
    
    ''' greedy layer-wise training '''
    epochs = 40

    iterations = int(60000*epochs/batch_size)

    dbn.train_greedylayerwise(vis_trainset=train_imgs, lbl_trainset=train_lbls, n_iterations=iterations)

    dbn.recognize(train_imgs, train_lbls)
    
    dbn.recognize(test_imgs, test_lbls)

    for digit in range(10):
       digit_1hot = np.zeros(shape=(1,10))
       digit_1hot[0,digit] = 1
       dbn.generate(digit_1hot, name="rbms")
