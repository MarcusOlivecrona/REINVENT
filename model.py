#!/usr/bin/env python
from __future__ import division
import numpy as np
import tensorflow as tf
import pickle
import time
import os, sys
import random

from data_structs import MolData, Vocabulary, tokenize
import scoring_functions

class Logger(object):
    """Class for printing to stdout as well as writing to a log file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

class REINVENT(object):
    def __init__(self, sess, config):

        self.config = config
        self.LEARNING_RATE = self.config['LEARNING_RATE']
        self.BATCH_SIZE = self.config['BATCH_SIZE']
        self.NUM_STEPS = self.config['NUM_STEPS']

        #For Prior
        self.NUM_EPOCHS = self.config['NUM_EPOCHS']

        #For Agent
        self.objective = self.config['AGENT_OBJECTIVE']
        self.sigma = self.config['sigma']
        self.sess = sess

        with open(self.config['VOCABULARY_PATH'], 'rb') as f:
		self.voc = pickle.load(f)

        self.voc.max_length = self.config['MAX_LENGTH']
        self.MAX_LENGTH = self.voc.max_length

        self.save_folder_path = self.config['SAVE_FOLDER_PATH']
        self.model_checkpoint = self.config['MODEL_CHECKPOINT_PATH']

        os.makedirs(self.save_folder_path, 0755)
        os.mkdir(self.save_folder_path + '/saved_model', 0755)
        sys.stdout = Logger(self.save_folder_path + '/log')

        #Create var_scopes from start so that we can refer to them consistenly 
        #whether or not its the first time they are used later on
        with tf.variable_scope("g_rnn") as self.var_scope_g: pass
        with tf.variable_scope("p_rnn") as self.var_scope_p: pass


    def pretrain_rnn(self):
        targets = tf.placeholder(tf.float32, [self.BATCH_SIZE, None, self.voc.vocab_size])
        learning_rate = tf.placeholder(tf.float32) 
        sequence_length = tf.placeholder(shape=[self.BATCH_SIZE], dtype=tf.int32)

        """Prepend 'GO' to inputs, means that 'EOS' will not be used as input 
        since we only unroll for [sequence_length] timesteps for each example"""
        inputs = self._prepend_start_token(targets)

        with tf.variable_scope(self.var_scope_g):
            logits = self._rnn(inputs, sequence_lengths=sequence_length, sample=False)

        start_token = tf.zeros([self.BATCH_SIZE, 1, self.voc.vocab_size])
        start_token = self._prepend_start_token(start_token)

        with tf.variable_scope(self.var_scope_g, reuse=True):
            gen_smiles, _ = self._rnn(start_token, sample=True)

        predicate = tf.equal(targets, 1.0)
        seq_loss = tf.where(predicate, logits, tf.ones_like(logits))
                                
        seq_loss = tf.reduce_sum(tf.reduce_sum(tf.log(seq_loss), axis=2), axis=1)
        seq_loss_per_step = tf.div(seq_loss, tf.to_float(sequence_length))

        seq_loss = -tf.reduce_mean(seq_loss)
        seq_loss_per_step = -tf.reduce_mean(seq_loss_per_step)

        ##################### Clip gradients ####################################
        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
	gvs = optimizer.compute_gradients(seq_loss, var_list=self.g_vars)
	capped_gvs = [(tf.clip_by_value(grad, -3., 3), var) for grad, var in gvs]
	train_op = optimizer.apply_gradients(capped_gvs)
        #########################################################################

        with open(self.save_folder_path + '/gen_smiles', 'wb') as f:
                    f.write('Generated smiles:\n\n')
        with open(self.save_folder_path + '/pretraining_losses', 'wb') as f:
                    f.write('Pretraining loss log \n\n')

        saver = tf.train.Saver()   
        self.sess.run(tf.global_variables_initializer())

        with open(self.config['MOL_DATA_PATH'], 'rb') as f:
		self.train_data = pickle.load(f)

        #Ensure the voc used to construct train data and decode to smiles are the same
        self.train_data.voc = self.voc
        
        print 'Pretraining started...'

        for epoch in range(self.NUM_EPOCHS):
        #Epoch used loosely here to refer to NUM_STEPS steps. An actual epoch is a lot more!
                epoch_start_time = time.time()
                for step in range(self.NUM_STEPS):
                    fetches = [seq_loss_per_step, train_op]
                    real_examples, seq_lens = self.train_data.sample(self.BATCH_SIZE) 

                    feed_dict = {targets:real_examples, 
                                 sequence_length:seq_lens, 
                                 learning_rate:self.LEARNING_RATE}
                                 
                    step_start_time = time.time()
                    loss, _ = self.sess.run(fetches=fetches, feed_dict=feed_dict)

                    print 'Loss: ' + str(round(loss, 5))

                    step_time_taken = time.time() - step_start_time

                    with open(self.save_folder_path + '/pretraining_losses', 'a') as f:
                       f.write('Epoch {} step {} loss: {:.3f}   time taken {:.2f}\n'.format(
                                                         epoch, step, loss, step_time_taken))

                self.LEARNING_RATE *= 0.98
                gen_smiles_ = self.sess.run(fetches=gen_smiles)

                print '\n' + '"'*50
                print 'Examples of training SMILES'
                print '"'*50
                for mol in real_examples[:5]:
                    print self.voc.decode(mol)
                print '"'*50 + '\n'

	        print '\n' + '"'*50
	        print 'Examples of generated SMILES'
	        print '"'*50

                epoch_time_taken = time.time() - epoch_start_time
                with open(self.save_folder_path + '/gen_smiles', 'a') as f:

		    f.write('\nPretraining epoch ' + str(epoch) + ' time taken ' +  
                                                       str(epoch_time_taken) + '\n')
		    for i, mol in enumerate(gen_smiles_):
                        f.write(self.voc.decode(mol) + '\n')    
			if i<5:  
                            print self.voc.decode(mol)

                print '"'*50 + '\n'
                saver.save(sess, self.save_folder_path + "/saved_model/model.ckpt")

    def sample(self, number_batches, savepath):
        inputs = tf.zeros([self.BATCH_SIZE, 1, self.voc.vocab_size])
        inputs = self._prepend_start_token(inputs)

        #Load from checkpoint, or if already loaded reuse varibles
        with tf.variable_scope(self.var_scope_g) as scope:
            try:
                gen_smiles, _ = self._rnn(inputs, sample=True)
                saver = tf.train.Saver()   
                saver.restore(sess, self.model_checkpoint)
            except ValueError:
                scope.reuse_variables()
                gen_smiles, _ = self._rnn(inputs, sample=True)

        with open(self.save_folder_path + '/' + savepath, 'wb') as f:
                    f.write('Generated smiles:\n\n')

        print "Sampling from model..."
        for i in range(number_batches):
            gen_smiles_ = self.sess.run(fetches=gen_smiles)
            with open(self.save_folder_path + '/' + savepath, 'a') as f:
                for i, mol in enumerate(gen_smiles_):
                    f.write(self.voc.decode(mol) + '\n')    

    def prior_likelihood(self, smiles):
        smiles = tokenize(smiles)
        seq_len = len(smiles) 
        target = self.voc.encode(smiles, seq_len=seq_len)
        target = tf.convert_to_tensor(target, dtype=tf.float32)
        target = tf.reshape(target, [1, tf.shape(target)[0], tf.shape(target)[1]])

        #Load from checkpoint, or if already loaded reuse varibles
        with tf.variable_scope(self.var_scope_g) as scope:
            try:
                prior_likelihood, logits = self._prior_likelihood(target)
                saver = tf.train.Saver()   
                saver.restore(sess, self.model_checkpoint)
            except ValueError:
                scope.reuse_variables()
                prior_likelihood, logits = self._prior_likelihood(target)

        #Probably shouldnt call them logits before they are actually log!
        logits = tf.log(logits)
        prior_likelihood_, logits_ = self.sess.run(fetches=[prior_likelihood, logits])

        chars = ['C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'F', 
                 'L', '(', ')', '1', '2', '3', '4', '=', 'EOS']

        for i in range(np.shape(logits_)[1]):
            for j, char in enumerate(chars):
                print "Step {:3d} Character {:2d} probability {:.3f}".format(
                                    i, j, logits_[0, i, self.voc.vocab[char]])

        print "Prior probability of {}: {:.2f}".format(smiles, prior_likelihood_[0])

    def train_agent(self):            

        if self.objective == "no_sulphur":
            scoring_function = scoring_functions.no_sulphur
        elif self.objective == "tanimoto":
            scoring_function = scoring_functions.tanimoto
        elif self.objective == "activity_model":
            #The model takes time to load, so do it just once rather than on every call
            self.activity_model = scoring_functions.restore_activity_model()
            scoring_function = scoring_functions.activity_model(self.activity_model)

        inputs = tf.zeros([self.BATCH_SIZE, 1, self.voc.vocab_size])
        inputs = self._prepend_start_token(inputs)

        #Load from checkpoint, or if already loaded reuse varibles
        with tf.variable_scope(self.var_scope_g) as scope:
            try:
                gen_smiles, agent_likelihood = self._rnn(inputs, sample=True)
                saver = tf.train.Saver()   
                saver.restore(sess, self.model_checkpoint)
            except ValueError:
                scope.reuse_variables()
                gen_smiles, agent_likelihood = self._rnn(inputs, sample=True)

        with tf.variable_scope(self.var_scope_p):
            prior_likelihood, _ = self._prior_likelihood(gen_smiles)

        score = self._score(gen_smiles, scoring_function)
        augmented_likelihood = prior_likelihood +  score * self.sigma
        reward = -tf.square(augmented_likelihood - agent_likelihood) 
        loss = tf.reduce_mean(-reward)

        ########################## Clip gradients #################################
        temp = set(tf.global_variables())
        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
	optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.LEARNING_RATE)
	gvs = optimizer.compute_gradients(loss, var_list=self.g_vars)
	capped_gvs = [(tf.clip_by_value(grad, -3., 3), var) for grad, var in gvs]
	train_op = optimizer.apply_gradients(capped_gvs)
        sess.run(tf.variables_initializer(set(tf.global_variables()) - temp))     
        ##########################################################################

        ############## Restore parameters from saved model #######################
        #Restore variables to a new name so that we can keep them fixed during
        #Training of the Agent
        prior_variables_name_mapping = dict()
        for g_var in [var for var in tf.trainable_variables() if "g_" in var.name]:
            for p_var in [var for var in tf.trainable_variables() if "p_" in var.name]:
                if p_var.name.replace("p_", "g_")==g_var.name:
                    prior_variables_name_mapping[g_var.name.rstrip(":0")] = p_var
        local_saver = tf.train.Saver(prior_variables_name_mapping)
        local_saver.restore(sess, self.model_checkpoint)
        #########################################################################

        for step in range(self.NUM_STEPS):
            start_time = time.time()
            score_, gen_smiles_, agent_likelihood_, prior_likelihood_, \
            augmented_likelihood_, reward_, loss_,  _ = self.sess.run(fetches=[score, 
                                                                            gen_smiles, 
                                                                            agent_likelihood, 
                                                                            prior_likelihood,
                                                                            augmented_likelihood,
                                                                            reward,
                                                                            loss,
                                                                            train_op])

            finish_time = time.time()
            print "\n\nStep {} out of {} generated smiles:\n".format(step, self.NUM_STEPS)
            for i, mol in enumerate(gen_smiles_):
                if i<10:
                    print self.voc.decode(mol)    
            print '\n'
            print "Total minibatch score: {:.2f}  Loss {:3.2f}  Time taken: {:.2f} seconds".format(
                                                      np.sum(score_), loss_, finish_time - start_time)
            print '\n'
            for i in range(10):
                print "Agent: {:4.2f} Prior: {:4.2f} Target: {:4.2f} Reward: {:4.2f}".format(
                                                                            agent_likelihood_[i], 
                                                                            prior_likelihood_[i], 
                                                                            augmented_likelihood_[i], 
                                                                            score_[i])
        print "Saving model..."
        local_saver = tf.train.Saver([var for var in tf.trainable_variables() if "g_" in var.name])
        local_saver.save(sess, self.save_folder_path + "/saved_model/model.ckpt")

    def _rnn(self, inputs, init_cell_state=None, sequence_lengths=None, sample=True): 

            softmax_w = tf.get_variable("softmax_w", [1024, self.voc.vocab_size], \
                                        initializer=tf.random_normal_initializer(0.5, 0.05))
            softmax_b = tf.get_variable("softmax_b", [self.voc.vocab_size], \
                                        initializer=tf.constant_initializer(0.1))
            
            cell = tf.contrib.rnn.GRUCell(1024)
            cell = tf.contrib.rnn.MultiRNNCell([cell]*3)

            inputs = tf.transpose(inputs, perm=[1,0,2]) #transpose to time-major
            inputs_ta = tf.TensorArray(dtype=tf.float32, size=self.MAX_LENGTH + 1)
            inputs_ta = inputs_ta.unstack(inputs)

            if sequence_lengths is None:
                sequence_lengths = self.MAX_LENGTH

            def get_loop(softmax_w, softmax_b):
                """Function to process inputs and outputs for RNN, depending on
                   whether we are getting log likelihood for target sequence or
                   sampling a new one. See tf.raw_rnn for documentation.
                """

                def train_loop(time, cell_output, cell_state, loop_state):
                    if cell_output is None: 
                        next_cell_state = cell.zero_state(tf.shape(inputs)[1], tf.float32)
                        emit_output = tf.zeros([self.voc.vocab_size], dtype=tf.float32)
                    else:
                        next_cell_state = cell_state
                        emit_output = tf.nn.softmax(tf.matmul(cell_output, softmax_w) + softmax_b) 
                    elements_finished = (time >= sequence_lengths)
                    finished = tf.reduce_all(elements_finished)
                    next_input = tf.cond(
                      finished,
                      lambda: tf.zeros([self.BATCH_SIZE, self.voc.vocab_size], dtype=tf.float32),
                      lambda: inputs_ta.read(time))
                    next_loop_state = None
                    return (elements_finished, next_input, next_cell_state,
                          emit_output, next_loop_state)

                def sample_loop(time, cell_output, cell_state, loop_state):
                    if cell_output is None:  # time == 0
                        if init_cell_state==None:
                            next_cell_state = cell.zero_state(self.BATCH_SIZE, tf.float32)
                        else:
                            next_cell_state = tuple(tf.unstack(init_cell_state, axis=0))
                        emit_output = tf.zeros([self.voc.vocab_size], dtype=tf.float32) 
                        cell_output = inputs_ta.read(time) 
                        elements_finished = tf.equal(tf.argmax(cell_output, axis=1), self.voc.vocab['EOS'])
                    else:
                        next_cell_state = cell_state
                        cell_output = tf.matmul(cell_output, softmax_w) + softmax_b 
                        output_sampling = tf.squeeze(tf.multinomial(cell_output, 1))
                        cell_output_softmax = tf.nn.softmax(cell_output)
                        elements_finished = tf.equal(output_sampling, self.voc.vocab['EOS'])
                        cell_output = tf.one_hot(output_sampling, depth=self.voc.vocab_size)
                        emit_output = tf.where(tf.equal(cell_output, 0), \
                                      tf.zeros_like(cell_output), cell_output_softmax)
                        elements_finished = tf.logical_or(elements_finished, (time>=sequence_lengths))
                    finished = tf.reduce_all(elements_finished)
                    next_input = tf.cond(
                      finished,
                      lambda: tf.zeros([cell_output.get_shape()[0], self.voc.vocab_size], dtype=tf.float32),
                      lambda: cell_output)
                    next_loop_state = None
                    return (elements_finished, next_input, next_cell_state,
                          emit_output, next_loop_state)

                if sample:
                    return sample_loop
                else:
                    return train_loop

            loop = get_loop(softmax_w, softmax_b)
             
            outputs_ta, final_state, _ = tf.nn.raw_rnn(cell, loop)
	    outputs = outputs_ta.stack()            

            #transform from time-major to batch-major
	    outputs = tf.transpose(outputs, perm=[1,0,2])

            if sample:
                probabilities = tf.reduce_max(outputs, axis=2)
                probabilities = tf.where(tf.equal(probabilities, 0), tf.ones_like(probabilities), probabilities)
                probabilities = tf.reduce_sum(tf.log(probabilities), axis=1)
                predictions = tf.where(tf.equal(outputs, 0), tf.zeros_like(outputs), tf.ones_like(outputs))
                return predictions, probabilities
            else:
                return outputs

    def _prior_likelihood(self, targets):
        inputs = self._prepend_start_token(targets)
        sequence_length = tf.to_int32(tf.reduce_sum(tf.reduce_sum(targets, axis=2), axis=1))
        logits = self._rnn(inputs, sequence_lengths=sequence_length, sample=False)

        # Get probability of generated samples from the pretrained model
        predicate = tf.equal(targets, 1.0)
        prior_likelihood = tf.where(predicate, logits, tf.ones_like(logits))
        prior_likelihood = tf.reduce_sum(tf.log(tf.reduce_prod(prior_likelihood, axis=2)), axis=1)
        return prior_likelihood, logits

    def _prepend_start_token(self, targets):
        targets_dims = tf.shape(targets)
        start_token = np.zeros([self.voc.vocab_size], dtype=np.float32)
        start_token[self.voc.vocab['GO']] = 1
        start_token = tf.convert_to_tensor(start_token)
        start_token = start_token * tf.ones([targets_dims[0], 1, self.voc.vocab_size])
        inputs = tf.concat([start_token, targets], 1)
        return inputs

    def _score(self, inputs, scoring_function):
        """Register the scoring function as a Tensorflow operation and define gradients"""

        def wrapper(inputs, scoring_function):
            def wrapped_scoring_function(inputs):
                inputs = [self.voc.decode(mol) for mol in inputs]
                score = scoring_function(inputs)
                return score
            return wrapped_scoring_function

        wrapped_scoring_function = wrapper(inputs, scoring_function)

        def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
            rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
            tf.RegisterGradient(rnd_name)(grad)
            g = tf.get_default_graph()
            with g.gradient_override_map({"PyFunc": rnd_name}):
                return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

        def _score_inner(inputs, name=None):
            with tf.name_scope(name, "score", [inputs]) as name:
                 pyfunc = py_func(wrapped_scoring_function,
                                [inputs],
                                [tf.float32],
                                name=name,
                                grad=_score_grads) 
                 return pyfunc[0]

        def _score_grads(op, grad):
            return grad 

        return _score_inner(inputs)

if __name__ == "__main__":

    model_config = {
              'LEARNING_RATE' : 0.0005,
              'NUM_STEPS' : 100,
              'NUM_EPOCHS' : 5,
              'BATCH_SIZE': 128,
              'MAX_LENGTH' : 140,
              'MODEL_CHECKPOINT_PATH' : 'saved_models/canonical_prior/model.ckpt',
              'MOL_DATA_PATH' : 'data/prior_trainingset_MolData',
              'VOCABULARY_PATH' : 'data/prior_trainingset_Voc',
              'SAVE_FOLDER_PATH' : 'saved_runs/' + 'run_' + time.strftime(
                                     "%Y_%m_%d_%H:%M:%S", time.localtime()),
              'sigma' : 3,
              'AGENT_OBJECTIVE' : 'activity_model', #"no_sulfur", "activity_model", "tanimoto"
              }

    print 'REINVENT started running...'
    with tf.Session() as sess:
        tf.set_random_seed(8)
        random.seed(8)
        model = REINVENT(sess, model_config)
        #model.pretrain_rnn()
        #model.prior_likelihood("COc1ccccc1N1CCN(CCCCNC(=O)c2ccccc2I)CC1")
        model.sample(10, savepath="gen_mols_before")
        model.train_agent()
        model.sample(10, savepath="gen_mols_after")
