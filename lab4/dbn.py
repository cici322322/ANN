from util import *
from rbm import RestrictedBoltzmannMachine

class DeepBeliefNet():    

    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''
    
    def __init__(self, sizes, image_size, n_labels, batch_size):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """

        self.rbm_stack = {
            
            'vis--hid' : RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                    is_bottom=True, image_size=image_size, batch_size=batch_size),
            
            'hid--pen' : RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"], batch_size=batch_size),
            
            'pen+lbl--top' : RestrictedBoltzmannMachine(ndim_visible=sizes["pen"]+sizes["lbl"], ndim_hidden=sizes["top"],
                                                        is_top=True, n_labels=n_labels, batch_size=batch_size)
        }
        
        self.sizes = sizes

        self.image_size = image_size

        self.batch_size = batch_size
        
        self.n_gibbs_recog = 15
        
        self.n_gibbs_gener = 200
        
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000

        self.n_labels = n_labels
        
        return

    def recognize(self,true_img,true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """
        
        n_samples = true_img.shape[0]
        n_labels = true_lbl.shape[1]
        
        vis = true_img # visible layer gets the image data
        
        lbl = np.ones(true_lbl.shape)/10. # start the net by telling you know nothing about labels  

        # Bottom-up
        print("vis--hid")
        h1 = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)[1]

        print("hid--pen")
        h2 = self.rbm_stack["hid--pen"].get_h_given_v_dir(h1)[1]
        h2_and_labels = np.concatenate((h2,lbl),axis=1)      
        
        for it in range(self.n_gibbs_recog):
            print("pen+lbl--top")
            h3 = self.rbm_stack["pen+lbl--top"].get_h_given_v(h2_and_labels)[1]
            h2_and_labels = self.rbm_stack["pen+lbl--top"].get_v_given_h(h3)[1]
            pass

        predicted_lbl = h2_and_labels[:, -n_labels:]
            
        print ("accuracy = %.2f%%"%(100.*np.mean(np.argmax(predicted_lbl,axis=1)==np.argmax(true_lbl,axis=1))))
        
        return

    def generate(self,true_lbl,name):
        
        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """
        
        n_sample = true_lbl.shape[0]
        n_labels = true_lbl.shape[1]
        
        records = []        
        fig,ax = plt.subplots(1,1,figsize=(3,3))#,constrained_layout=True)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]); ax.set_yticks([])

        lbl = true_lbl # labels layer gets the label data
        h2 = sample_binary(0.5*np.ones((n_sample, self.sizes['pen']))) # random data at the beginning in layer 2 (pen)
        # h2 = np.ones((n_sample, self.sizes['pen']))
        h2_and_labels = np.concatenate((h2,lbl),axis=1)
            
        for _ in range(self.n_gibbs_gener):
            # firstly we go up-bottom to get the visible layer
            h1 = self.rbm_stack["hid--pen"].get_v_given_h_dir(h2)[1]
            vis = self.rbm_stack["vis--hid"].get_v_given_h_dir(h1)[1]

            # Secondly, we continue with the Gibbs-sampling in the top-layer
            h3 = self.rbm_stack["pen+lbl--top"].get_h_given_v(h2_and_labels)[1]
            h2_and_labels = self.rbm_stack["pen+lbl--top"].get_v_given_h(h3)[1]
            h2 = h2_and_labels[:, :-n_labels]

            
            records.append( [ ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True, interpolation=None) ] )
            
        anim = stitch_video(fig,records).save("%s.generate%d.html"%(name,np.argmax(true_lbl)))            
            
        return

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Greedy layer-wise training by stacking RBMs. This method first tries to load previous saved parameters of the entire RBM stack. 
        If not found, learns layer-by-layer (which needs to be completed) .
        Notice that once you stack more layers on top of a RBM, the weights are permanently untwined.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """


        try :

            self.loadfromfile_rbm(loc="trained_rbm",name="vis--hid")
            self.rbm_stack["vis--hid"].untwine_weights()            
            
            self.loadfromfile_rbm(loc="trained_rbm",name="hid--pen")
            self.rbm_stack["hid--pen"].untwine_weights()
            # In the last one weights are symmetric
            self.loadfromfile_rbm(loc="trained_rbm",name="pen+lbl--top")        

        except IOError :
        
            print ("training vis--hid")
            """ 
            CD-1 training for vis--hid 
            """            

            self.rbm_stack["vis--hid"].cd1(vis_trainset,n_iterations)

            self.savetofile_rbm(loc="trained_rbm",name="vis--hid")

            print ("training hid--pen")
            self.rbm_stack["vis--hid"].untwine_weights()            
            """ 
            CD-1 training for hid--pen 
            """            
            h1_trainset =  self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_trainset)[1]
            self.rbm_stack["hid--pen"].cd1(h1_trainset,n_iterations)

            self.savetofile_rbm(loc="trained_rbm",name="hid--pen")            

            print ("training pen+lbl--top")
            self.rbm_stack["hid--pen"].untwine_weights()
            """ 
            CD-1 training for pen+lbl--top 
            """

            h2_trainset_data = self.rbm_stack["hid--pen"].get_h_given_v_dir(h1_trainset)[1]
            h2_trainset = np.concatenate((h2_trainset_data,lbl_trainset),axis=1) # concatenate data from the previous layer and labels
            self.rbm_stack["pen+lbl--top"].cd1(h2_trainset,n_iterations)

            self.savetofile_rbm(loc="trained_rbm",name="pen+lbl--top")   


        return    

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        
        print ("\ntraining wake-sleep..")

        try :
            
            self.loadfromfile_dbn(loc="trained_dbn",name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn",name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn",name="pen+lbl--top")
            
        except IOError :            

            self.n_samples = vis_trainset.shape[0]
            n_samples = vis_trainset.shape[0]
            n_labels = lbl_trainset.shape[1]
            
            for it in range(n_iterations):  

                minibatch_ndx = int(it % (n_samples/self.batch_size))
                minibatch_end = min([(minibatch_ndx+1)*self.batch_size, n_samples])
                minibatch = vis_trainset[minibatch_ndx*self.batch_size:minibatch_end, :]  
                minibatch_lbl = lbl_trainset[minibatch_ndx*self.batch_size:minibatch_end, :] 
                                
                """ 
                wake-phase : drive the network bottom-to-top using visible and label data
                """

                vis = minibatch
                h1 = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis)[1]

                h2 = self.rbm_stack["hid--pen"].get_h_given_v_dir(h1)[1]

                h2_and_labels = np.concatenate((h2,minibatch_lbl),axis=1)
                h3 = self.rbm_stack["pen+lbl--top"].get_h_given_v(h2_and_labels)[1]

                """
                alternating Gibbs sampling in the top RBM : also store neccessary information for learning this RBM
                """
                # Predictions neccesary for updating generation matrix

                v_0_top = h2_and_labels
                h_0_top = h3

                for it in range(self.n_gibbs_wakesleep):
                    h2_and_labels = self.rbm_stack["pen+lbl--top"].get_v_given_h(h3)[1]
                    h3 = self.rbm_stack["pen+lbl--top"].get_h_given_v(h2_and_labels)[1]
                    pass

                v_k_top = h2_and_labels
                h_k_top = h3

                h2_down = h2_and_labels[:, :-n_labels]
                h1_down = self.rbm_stack["hid--pen"].get_v_given_h_dir(h2_down)[1]
                vis_down = self.rbm_stack["vis--hid"].get_v_given_h_dir(h1_down)[1]


                """
                sleep phase : from the activities in the top RBM, drive the network top-to-bottom
                """

                """
                predictions : compute generative predictions from wake-phase activations, 
                              and recognize predictions from sleep-phase activations
                """
                vis_pred_gen = self.rbm_stack["vis--hid"].get_v_given_h_dir(h1)[1]
                h1_pred_gen = self.rbm_stack["hid--pen"].get_v_given_h_dir(h2)[1]

                h2_pred_rec = self.rbm_stack["hid--pen"].get_h_given_v_dir(h1_down)[1]
                h1_pred_rec = self.rbm_stack["vis--hid"].get_h_given_v_dir(vis_down)[1]
                
                """ 
                update generative parameters :
                here you will only use "update_generate_params" method from rbm class
                """

                self.rbm_stack["vis--hid"].update_generate_params(h1,vis,vis_pred_gen)
                self.rbm_stack["hid--pen"].update_generate_params(h2,h1,h1_pred_gen)

                """ 
                update parameters of top rbm:
                here you will only use "update_params" method from rbm class
                """

                self.rbm_stack["pen+lbl--top"].update_params(v_0_top,h_0_top,v_k_top,h_k_top)
                
                """ 
                update recognition parameters :
                here you will only use "update_recognize_params" method from rbm class
                """

                self.rbm_stack["hid--pen"].update_recognize_params(h1_down,h2_down,h2_pred_rec)
                self.rbm_stack["vis--hid"].update_recognize_params(vis_down,h1_down,h1_pred_rec)

                if it % self.print_period == 0 : print ("iteration=%7d"%it)
                        
            self.savetofile_dbn(loc="trained_dbn",name="vis--hid")
            self.savetofile_dbn(loc="trained_dbn",name="hid--pen")
            self.savetofile_rbm(loc="trained_dbn",name="pen+lbl--top")            

        return

    
    def loadfromfile_rbm(self,loc,name):
        
        self.rbm_stack[name].weight_vh = np.load("%s/rbm.%s.weight_vh.npy"%(loc,name))
        self.rbm_stack[name].bias_v    = np.load("%s/rbm.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h    = np.load("%s/rbm.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_rbm(self,loc,name):
        
        np.save("%s/rbm.%s.weight_vh"%(loc,name), self.rbm_stack[name].weight_vh)
        np.save("%s/rbm.%s.bias_v"%(loc,name),    self.rbm_stack[name].bias_v)
        np.save("%s/rbm.%s.bias_h"%(loc,name),    self.rbm_stack[name].bias_h)
        return
    
    def loadfromfile_dbn(self,loc,name):
        
        self.rbm_stack[name].weight_v_to_h = np.load("%s/dbn.%s.weight_v_to_h.npy"%(loc,name))
        self.rbm_stack[name].weight_h_to_v = np.load("%s/dbn.%s.weight_h_to_v.npy"%(loc,name))
        self.rbm_stack[name].bias_v        = np.load("%s/dbn.%s.bias_v.npy"%(loc,name))
        self.rbm_stack[name].bias_h        = np.load("%s/dbn.%s.bias_h.npy"%(loc,name))
        print ("loaded rbm[%s] from %s"%(name,loc))
        return
        
    def savetofile_dbn(self,loc,name):
        
        np.save("%s/dbn.%s.weight_v_to_h"%(loc,name), self.rbm_stack[name].weight_v_to_h)
        np.save("%s/dbn.%s.weight_h_to_v"%(loc,name), self.rbm_stack[name].weight_h_to_v)
        np.save("%s/dbn.%s.bias_v"%(loc,name),        self.rbm_stack[name].bias_v)
        np.save("%s/dbn.%s.bias_h"%(loc,name),        self.rbm_stack[name].bias_h)
        return
    