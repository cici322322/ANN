import numpy as np
import copy, random

class network:
	def __init__(self, dim, sync = True, data = None, verbose = False, show_gap = None,
				show_handle = None, show_delay = None, activity = 0, bias = 0,
				diagonal = 1):
		self.dim = dim
		self.sync = sync
		self.activity = activity
		self.bias = bias
		self._set_weight = False
		self._verbose = verbose
		self._max_iter = 1000
		self._show_gap = show_gap
		self._show_handle = show_handle
		self._trace = [[], []]
		self._diagonal = diagonal
		self._show_delay = show_delay
		assert not (self._show_gap != None and self._show_handle == None)
		if data != None:
			assert isinstance(data, np.ndarray)
			self.update_weight(data)

	@property
	def trace(self):
		return self._trace

	def lazy_update_weight(self, data):
		assert isinstance(data, np.ndarray)
		assert data.shape[0] == self.dim
		assert self._set_weight
		# self.w = np.zeros((self.dim, self.dim))
		vector_data = np.reshape(data, (1, self.dim))
		for i in range(data.shape[0]):
			self.w = self.w + np.dot(vector_data.T - self.activity, vector_data - self.activity) / self.dim
		

	def update_weight(self, data):
		assert isinstance(data, np.ndarray)
		assert data.shape[1] == self.dim
		self.w = np.zeros((self.dim, self.dim))
		for i in range(data.shape[0]):
			self.w = self.w + np.dot(data[i:i + 1].T - self.activity, data[i:i + 1] - self.activity)
		self.w = self.w / self.dim
		for i in range(self.dim):
			self.w[i][i] *= self._diagonal
		self._set_weight = True
		if self._verbose:
			print (self.w)

	def update_weight_zero(self):
		self.w = np.zeros((self.dim, self.dim))
		self._set_weight = True

	def update_weight_normal(self):
		self.w = np.zeros((self.dim, self.dim))
		for i in range(self.dim):
			for j in range(self.dim):
				self.w[i][j] = random.normalvariate(0, 5)
		self._set_weight = True

	def update_weight_symmetry(self):
		self.w = np.zeros((self.dim, self.dim))
		for i in range(self.dim):
			for j in range(self.dim):
				self.w[i][j] = random.normalvariate(0, 5)
		self.w = 0.5 * (self.w + self.w.T)
		assert np.array_equal(self.w, self.w.T)
		self._set_weight = True


	def update_state(self, init_state):
		assert self._set_weight
		assert init_state.shape == (self.dim,)
		if self.sync:
			return self._sync_update_state(init_state)
		else:
			return self._unsync_update_state(init_state)

	def stationary_point(self, init_state):
		if np.array_equal(self._sign_list(np.dot(init_state, self.w)), init_state):
			return True
		else:
			return False

	def get_energy(self, init_state):
		vector_state = np.reshape(init_state, (1, init_state.shape[0]))
		energy_matrix = np.multiply(self.w, np.dot(vector_state.T, vector_state))
		return -np.sum(np.sum(energy_matrix, axis = 1))

	def binary_stationary_point(self, init_state):
		if np.array_equal(self._sign_binary_list(np.dot(init_state, self.w) - self.bias), init_state):
			return True
		else:
			return False

	def _sign_binary_scala(self, value):
		if value > 0:
			return 1.
		else:
			return 0.

	def _sign_binary_list(self, state):
		for i in range(self.dim):
			state[i] = self._sign_binary_scala(state[i])
		return state

	def _sign_scala(self, value):
		if value > 0:
			return 1.
		elif value == 0:
			return 0.
		else:
			return -1.

	def _sign_list(self, state):
		for i in range(self.dim):
			state[i] = self._sign_scala(state[i])
		return state

	def _sync_update_state(self, init_state):
		if self._verbose:
			print ("------------state debug-----------")

		old_state = copy.deepcopy(init_state)
		for i in range(self._max_iter):
			new_state = self._sign_list(np.dot(old_state, self.w))

			if self._verbose:
				print ("[Verbose Iteration {}]".format(i + 1))
				print (np.dot(old_state, self.w))
				print (new_state)

			if np.array_equal(new_state, old_state):
				
				if self._verbose:
					print ("[Debug] Converge in {} Iterations".format(i))
					print ("-------------debug end-------------")

				return i, old_state
			old_state = copy.deepcopy(new_state)
		
		print ("[Warning] Can't Converge in {} Iterations".format(self._max_iter))

		if self._verbose:
			print ("-------------debug end-------------")

		return -1, new_state

	def _unsync_update_state(self, init_state):
		show_count = 0
		self._trace = [[], []]
		new_state = copy.deepcopy(init_state)
		for i in range(self._max_iter):
			old_state = copy.deepcopy(new_state)
			index = random.sample(range(self.dim), self.dim)
			for j in range(self.dim):
				new_state[index[j]] = self._sign_scala(np.dot(new_state, self.w[index[j]]))
				if (self._show_gap != None) and (show_count % self._show_gap == 0):
					self._trace[0].append(show_count)
					self._trace[1].append(self.get_energy(new_state))
					print ("Energy = {}".format(self.get_energy(new_state)))
					self._show_handle(new_state, self._show_delay)
				show_count += 1
			if np.array_equal(new_state, old_state):
				return i, old_state
				
		print ("[Warning] Can't Converge in {} Iterations".format(self._max_iter))
		return -1, new_state
