reward = 0
def insertionSort(alist):
   for index in range(1,len(alist)):

     currentvalue = alist[index]
     position = index
     flag = False

     while position>0 and alist[position-1]>currentvalue:
         alist[position]=alist[position-1]
         position = position-1
         flag = True

     alist[position]=currentvalue
     global reward
     if (flag == True):
     	reward = reward -1
     	print(alist)

alist = [ 7, 4, 2, 5, 3, 6, 1]
insertionSort(alist)
print(reward)