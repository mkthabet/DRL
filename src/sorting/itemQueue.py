import numpy as np
import random


class ItemQueue:
	def __init__(self, num_items):
		self.num_items = num_items
		self.act_space_size = self.num_items
		

	def reset(self, init_list = []):
		if len(init_list) == 0:
			self.item_list = [[i+1] for i in range(self.num_items)]
			random.shuffle(self.item_list)
		else:
			self.item_list = [[i] for i in init_list]
			num_items = len(init_list)
		self.item_picked = [0];
		#the 1st element in the state vector is the id of the item picked up, or 0 if no item is picked up.
		return self._generateState()


	def step(self, action):
		#action {1, ..., n-1, n} where n=num_items is a position in the list starting from 1.
		#positions start from 0, item id's start from 1
		assert action<self.num_items and action>=0, "action cannot exceed number of items or be less than 0, action = %r" % action

		if self.item_picked==[0]:	#action is interpreted as picking up
			self.item_picked = self.item_list[action]
			self.item_list[action] = [0]
		else:	#action is inserting
			self.item_list.remove([0])
			self.item_list.insert(action, self.item_picked)
			self.item_picked = [0]

		done = 0
		reward = -1
		sorted_list = list(self.item_list)
		sorted_list.sort()
		if (sorted_list == self.item_list) and (self.item_picked == [0]):
			done = 1
			#print "episode done. Sorted."

		return self._generateState(), reward, done

	def _generateState(self):
		state = list(self.item_list)
		state.insert(0, self.item_picked)
		stateArr = np.array(state)
		return stateArr.flatten()

	def printState(self):
		state = self._generateState()
		stateArr = np.array(state)
		print "state = %s" % stateArr.T
		#x = stateArr.shape
		#print "state dims: "
		#print(x)
		#for i in self.item_list: print i
		#print self.item_list
		#print state
		#for i in state: print i

	def getStateSpaceSize(self):
		return self.num_items+1

	def getActSpaceSize(self):
		return self.act_space_size



# testList = ItemQueue(4)
# testList.reset()
# testList.printState()
# testList.step(1)
# testList.printState()
# testList.step(1)
# testList.printState()



