#iterators
nums = [1, 2, 3, 4]
obj = iter(nums)
print(next(obj))
print(next(obj))
print(next(obj))
print(next(obj))

#generator
def nums():
   for i in range(1, 5):
       yield i

obj = nums()
print(next(obj))
print(next(obj))
print(next(obj))
print(next(obj))

class Alphabets:
  def __iter__(self):
      self.val = 65
      return self

  def __next__(self):
      if self.val > 90:
          raise StopIteration
      temp = self.val
      self.val += 1
      return chr(temp)

my_letters = Alphabets()
my_iterator = iter(my_letters)
for letter in my_iterator:
   print(letter, end = " ")


def Alphabets():
   for i in range(65, 91):
       yield chr(i)

my_letters = Alphabets()

for letter in my_letters:
   print(letter, end=" ")


def gener():
   num = 1
   while True:
       yield num
       num += 1

obj = gener()
print(next(obj))
print(next(obj))
print(next(obj))


from collections.abc import Generator, Iterator
print(issubclass(Generator, Iterator))


def gener():
   List = ["orange", "green", "black"]
   for item in List:
       yield item

iter_obj = gener()
print(next(iter_obj))
print(next(iter_obj))
print(next(iter_obj))


def abcd():
   for i in range(97, 101):
       yield chr(i)


class Multiples:
  def __iter__(self):
      self.val = 1
      return self
  def __next__(self):
      temp = self.val
      self.val += 1
      return temp*5

multiples5 = Multiples()
obj = iter(multiples5)

print(next(obj))

def Multiples():
   i = 1
   while True:
       yield i*5
       i += 1

multiples5 = Multiples()
obj = multiples5
print(next(obj))
