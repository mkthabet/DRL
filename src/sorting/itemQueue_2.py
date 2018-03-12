import numpy as np
import random


class ItemQueue:
	def __init__(self, num_items):
		self.num_items = num_items
		self.act_space_size = self.num_items
		

	def reset(self, ordering_type, init_list = []):
		#ordering_type: 1 or 2 for ascending or descending on 1st property, 3 or 4 for 2nd property
		self.ordering_type = ordering_type
		if len(init_list) == 0:
			list_1 = np.array([[i+1] for i in range(self.num_items)]).flatten()
			list_2 = list_1.copy()
			random.shuffle(list_1)
			random.shuffle(list_2)
			#self.item_list = np.array([[list_1[i],list_2[i]] for i in range(self.num_items)])
			self.item_list = np.array(zip(list_1,list_2))
			#print np.asarray(self.item_list)

		else:
			self.item_list = np.array(init_list)
			num_items = len(init_list)
		self.item_picked = np.array([0,0]);
		#print self.item_picked
		#the 1st element in the state vector is the id of the item picked up, or 0 if no item is picked up.
		return self._generateState()


	def step(self, action):
		#action {1, ..., n-1, n} where n=num_items is a position in the list starting from 1.
		#positions start from 0, item id's start from 1
		assert action<self.num_items and action>=0, "action cannot exceed number of items or be less than 0, action = %r" % action

		if self.item_picked==np.array([0,0]):	#action is interpreted as picking up
			self.item_picked = np.array(self.item_list[action])
			self.item_list[action] = np.array([0,0])
		else:	#action is inserting
			self.item_list.remove(np.array([0,0]))
			self.item_list.insert(action, self.item_picked)
			self.item_picked = np.array([0,0])

		done = 0
		reward = -1

		#check if the sorting is finished
		temp_list = np.array(self.item_list)
		temp_list.flatten()
		if self.ordering_type == 1:
			sorted_list = np.array([i for i in xrange(0,len(temp_list),2)])
			sorted_list.sort()
		elif self.ordering_type == 2:
			sorted_list = np.array([i for i in xrange(0,len(temp_list),2)])
			sorted_list[::-1].sort()
		elif self.ordering_type == 3:
			sorted_list = np.array([i for i in xrange(1,len(temp_list),2)])
			sorted_list.sort()
		else:
			sorted_list = np.array([i for i in xrange(1,len(temp_list),2)])
			sorted_list[::-1].sort()
		if (sorted_list == temp_list) and (self.item_picked == [0,0]):
			done = 1
			#print "episode done. Sorted."

		return self._generateState(), reward, done

	def _generateState(self):
		temp = self.item_picked.copy()
		state = np.append(np.expand_dims(temp,0), self.item_list, axis=0)
		return state

	def printState(self):
		state = self._generateState()
		print(state)


	def getStateSpaceSize(self):
		return self.num_items+1

	def getActSpaceSize(self):
		return self.act_space_size


testList = ItemQueue(2)
testList.reset(ordering_type = 1)
testList.printState()
testList.step(1)
# testList.printState()
# testList.step(1)
# testList.printState()



