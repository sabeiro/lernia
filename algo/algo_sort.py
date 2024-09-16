import numpy as np

def bubble_sort(l):
  # Traverse (travel across) through every element on the list
  for i in range(0, len(l) - 1):
    # Compare each item in list 1 by 1. Comparison in each iteration 
    # will shorten as the last elements become sorted
    for j in range(0, len(l) - 1 - i):
      # traverse the list from 0 to n-i-1 
      # if the element found is greater than the next element, swap
      if l[j] > l[j + 1]:
        l[j], l[j + 1] = l[j + 1], l[j]
        
  return l

def insertionSort(l):
  # all the values after the first
  index_length = range(1, len(l))
  # to do an operation on all these values
  # for all the value is the index_length value,
  for i in index_length:
    # we want to sort those values
    sort = l[i]
    # while the item to the left is greater than the item
    # to the right
    # notice that we also have to write i > 0 bc python allows
    # for negative indexing
    while l[i-1] > sort and i > 0:
      # swap
      l[i], l[i-1] = l[i-1], l[i]
      # to continue doing comparisons down the list,
      # look at the next item
      i = i - 1
  return list

def recursive(n):
  # This is the base case we are working towards
  # When we get to where n==0, we return 1
  if n == 0:
    return 1
  # here we move towards the base case
  return n * recursive(n - 1)

def merge(left, right):
  elements = len(left) + len(right)
  merged_list = [0] * elements
  left_pointer = 0
  right_pointer = 0
  i = 0

  # while there are elements in either list
  while left_pointer < len(left) or right_pointer < len(right):
    # if there are elements in both lists
    if left_pointer < len(left) and right_pointer < len(right):
      if left[left_pointer] < right[right_pointer]:
        merged_list[i] = left[left_pointer]
        left_pointer += 1
      else:
        merged_list[i] = right[right_pointer]
        right_pointer += 1
      i += 1
      # if there are only elements in the left list
    elif left_pointer < len(left):
      merged_list[i] = left[left_pointer]
      left_pointer += 1
      i += 1
      # if there are only elements in the right list
    elif right_pointer < len(right):
      merged_list[i] = right[right_pointer]
      right_pointer += 1
      i += 1
  return merged_list
 
# sort function
def merge_sort(l):
  if len(l) <= 1:
    return l
  else:
    mid = (len(l)) // 2
    left_array = l[:mid]
    right_array = l[mid:]
    left_array = merge_sort(left_array)
    right_array = merge_sort(right_array)
    result = merge(left_array, right_array)
  return result
 
# in-place merge sort algorithm
def merge_in_place(l, begin, mid, end):
  begin_high = mid + 1  
  # if mid and begin_high are already in order, return
  if l[mid] <= l[begin_high]:
    return
  # while pointers are within bounds
  while begin <= mid and begin_high <= end:
    # if begin element is in order w/ respect to begin_high element
    if l[begin] <= l[begin_high]:
      # increment begin
      begin += 1
    else:
      # current value is at begin of begin_high
      value = l[begin_high]
      # index is begin_high
      index = begin_high
      
      # while index is not equal to begin
      while index != begin:
        # swap item at index with it's left-neighbor
        l[index] = l[index-1]
        # decrement index
        index -= 1
        # value at list begin is new value
        l[begin] = value
        # increment all pointers (minus end)
        begin += 1
        mid += 1
        begin_high += 1
 
# sorting in place
def merge_sort_in_place(l, left, right):
  if left < right:
    mid = (left + right) // 2  
    merge_sort_in_place(l, left, mid)
    merge_sort_in_place(l, mid+1, right)
    merge_in_place(l, left, mid, right)
  return l
 
def selection_sort(lst):
  """
    Selection sort function
    :param lst: List of integers
  """
  # Traverse through all lst elements
  for i in range(len(l)):
    # Find the minimum element in unsorted lst
    min_index = i
    for j in range(i + 1, len(l)):
      if l[min_index] > l[j]:
        min_index = j

  # Swap the found minimum element with the first element
  l[i], l[min_index] = l[min_index], l[i]

 

l = [5, 45, 22 , 3, 9, 0, 12, 6, 1]
print(sorted(l))
print(bubble_sort(l))
print(merge_sort_in_place(l, 0, len(l)-1))


# Python program for implementation of Selection
# Sort
import sys
 
 
A = [64, 25, 12, 22, 11]
 
# Traverse through all array elements
for i in range(len(A)):
     
    # Find the minimum element in remaining
    # unsorted array
    min_idx = i
    for j in range(i+1, len(A)):
        if A[min_idx] > A[j]:
            min_idx = j
             
    # Swap the found minimum element with
    # the first element    
    A[i], A[min_idx] = A[min_idx], A[i]
 
# Driver code to test above
print ("Sorted array")
for i in range(len(A)):
    print("%d" %A[i]),



# Python program for implementation of Bubble Sort
 
def bubbleSort(arr):
    n = len(arr)
 
    # Traverse through all array elements
    for i in range(n):
 
        # Last i elements are already in place
        for j in range(0, n-i-1):
 
            # traverse the array from 0 to n-i-1
            # Swap if the element found is greater
            # than the next element
            if arr[j] > arr[j+1] :
                arr[j], arr[j+1] = arr[j+1], arr[j]
 
# Driver code to test above
arr = [64, 34, 25, 12, 22, 11, 90]
 
bubbleSort(arr)
 
print ("Sorted array is:")
for i in range(len(arr)):
    print ("%d" %arr[i]),



# Python program for implementation of Insertion Sort
 
# Function to do insertion sort
def insertionSort(arr):
 
    # Traverse through 1 to len(arr)
    for i in range(1, len(arr)):
 
        key = arr[i]
 
        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i-1
        while j >= 0 and key < arr[j] :
                arr[j + 1] = arr[j]
                j -= 1
        arr[j + 1] = key
 
 
# Driver code to test above
arr = [12, 11, 13, 5, 6]
insertionSort(arr)
for i in range(len(arr)):
    print ("% d" % arr[i])

 Python program for implementation of MergeSort
def mergeSort(arr):
    if len(arr) > 1:
 
        # Finding the mid of the array
        mid = len(arr)//2
 
        # Dividing the array elements
        L = arr[:mid]
 
        # into 2 halves
        R = arr[mid:]
 
        # Sorting the first half
        mergeSort(L)
 
        # Sorting the second half
        mergeSort(R)
 
        i = j = k = 0
 
        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1
 
        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1
 
        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1
 
# Code to print the list
 
 
def printList(arr):
    for i in range(len(arr)):
        print(arr[i], end=" ")
    print()
 
 
# Driver Code
arr = [12, 11, 13, 5, 6, 7]
print("Given array is", end="\n")
printList(arr)
mergeSort(arr)
print("Sorted array is: ", end="\n")
printList(arr)



# Python3 implementation of QuickSort
 
# This Function handles sorting part of quick sort
# start and end points to first and last element of
# an array respectively
def partition(start, end, array):
     
    # Initializing pivot's index to start
    pivot_index = start
    pivot = array[pivot_index]
     
    # This loop runs till start pointer crosses
    # end pointer, and when it does we swap the
    # pivot with element on end pointer
    while start < end:
         
        # Increment the start pointer till it finds an
        # element greater than pivot
        while start < len(array) and array[start] <= pivot:
            start += 1
             
        # Decrement the end pointer till it finds an
        # element less than pivot
        while array[end] > pivot:
            end -= 1
         
        # If start and end have not crossed each other,
        # swap the numbers on start and end
        if(start < end):
            array[start], array[end] = array[end], array[start]
     
    # Swap pivot element with element on end pointer.
    # This puts pivot on its correct sorted place.
    array[end], array[pivot_index] = array[pivot_index], array[end]
     
    # Returning end pointer to divide the array into 2
    return end
     
# The main function that implements QuickSort
def quick_sort(start, end, array):
     
    if (start < end):
         
        # p is partitioning index, array[p]
        # is at right place
        p = partition(start, end, array)
         
        # Sort elements before partition
        # and after partition
        quick_sort(start, p - 1, array)
        quick_sort(p + 1, end, array)
         
# Driver code
array = [ 10, 7, 8, 9, 1, 5 ]
quick_sort(0, len(array) - 1, array)
 
print(f'Sorted array: {array}')



# Python3 program for implementation of Shell Sort
 
def shellSort(arr):
    gap = len(arr) // 2 # initialize the gap
 
    while gap > 0:
        i = 0
        j = gap
         
        # check the array in from left to right
        # till the last possible index of j
        while j < len(arr):
     
            if arr[i] >arr[j]:
                arr[i],arr[j] = arr[j],arr[i]
             
            i += 1
            j += 1
         
            # now, we look back from ith index to the left
            # we swap the values which are not in the right order.
            k = i
            while k - gap > -1:
 
                if arr[k - gap] > arr[k]:
                    arr[k-gap],arr[k] = arr[k],arr[k-gap]
                k -= 1
 
        gap //= 2
 
 
# driver to check the code
arr2 = [12, 34, 54, 2, 3]
print("input array:",arr2)
 
shellSort(arr2)
print("sorted array",arr2)
