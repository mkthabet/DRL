import numpy as np
import random


class Toh:
	def __init__(self, num_items, num_stacks):
		self.num_items = num_items
		self.num_stacks = num_stacks
		#assert num_stacks>=num_items, "number of items cannot exceed number of stacks"
		self.act_space_size = (self.num_stacks -1)*self.num_stacks
		

	def reset(self):
		self.stacks = np.zeros((self.num_stacks, self.num_items))
		self.stacks[0,:] = np.array([i+1 for i in range(self.num_items)])
		#print self.stacks
		return self._generateState()


	def step(self, action):
		#action {1, ..., n-1, n} where n=num_items is a position in the list starting from 1.
		#positions start from 0, item id's start from 1
		assert action<self.act_space_size and action>=0, "action value cannot exceed (num_stacks-1)*num_stacks or be negative, action = %r" % action

		dest_stack = action%(self.num_stacks-1)

		source_stack = action//(self.num_stacks-1)
		if source_stack <= dest_stack:
			dest_stack += 1
		#print "source stack" ,source_stack
		#print "dest stack" ,dest_stack

		if self.stacks[source_stack,0] != 0:
			self.stacks[dest_stack,:] = np.roll(self.stacks[dest_stack,:],1)
			self.stacks[dest_stack,0] = self.stacks[source_stack,0]
			self.stacks[source_stack,0] = 0.
			self.stacks[source_stack,:] = np.roll(self.stacks[source_stack,:],-1)
		else:	#penalize for doing nothing this step
			reward = -600
			#print "Meaningless action"
			done = 0
			return self._generateState(), reward, done

		done = 0
		reward = -1
		#check for completeion and rewards for them, or illegal moves and penalize for them
		for i in range(self.num_stacks):
			sorted_stack = np.copy(np.trim_zeros(self.stacks[i,:]))
			sorted_stack.sort()
			#print sorted_stack
			if not np.array_equal(sorted_stack, np.trim_zeros(self.stacks[i,:])):
				reward -= 600
				done = 1
				print "Illegal move."
				break
			if np.array_equal(sorted_stack, self.stacks[i,:]) and i != 0:
				reward += 500
				done = 1
				print "Done!########################################"

		#print self.stacks
		return self._generateState(), reward, done

	def _generateState(self):
		return self.stacks.flatten()

	def printState(self):
		print self.stacks 

	def getStateSpaceSize(self):
		return self.num_items*self.num_stacks

	def getActSpaceSize(self):
		return self.act_space_size



#testEnv = Toh(3,3)
#testEnv.reset()
#testEnv.step(0)
#testEnv.printState()
# testList.reset()
# testList.printState()
# testList.step(1)
# testList.printState()
# testList.step(1)
# testList.printState()



