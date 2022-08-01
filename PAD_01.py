#Homework PAD1
#Task 1

class animal:
    
    def __init__(self, genus, gender = 'Female'):
        self.isAlive = True
        self.gender = gender
        self.genus = genus

    def breed(self, partner):
            if (self.gender == 'Female' and partner.gender == 'Male' and self.genus == partner.genus):
                return self.__class__(self.genus, self.gender)
            else:
                raise Exception("attribute not found")

class cat(animal):
    @staticmethod
    def purr():
        return 'purr'

class dog(animal):
    @staticmethod
    def woof():
        return 'woof woof'

dog_1 = dog('Canis', 'Female')
dog_2 = dog('Canis', 'Male')

cat_1 = cat('Felis', 'Female')
cat_2 = cat('Felis', 'Male')


dog_1.breed(dog_2)
print(dog_1.breed(cat_1))
cat_1.breed(cat_2)


#Task 2

old_group_sum = 0
old_group_count = 0
young_group_sum = 0



class worker:

    salary_sum = 0
    workers_count = 0
    _registry =[]

    def __init__(self, number, name, age, salary):
        self.number = number
        self.name = name
        self.age = 2021 - age
        self.salary = salary
        type(self).salary_sum += salary
        type(self).workers_count += 1
        type(self).avg_pay = self.salary_sum/self.workers_count
        self._registry.append(self)


worker_1 = worker(1, 'Adam', 1983, 1500)
worker_2 = worker(2, 'Anna', 1981, 1700)
worker_3 = worker(3, 'Błażej', 1990, 1800)
worker_4 = worker(4, 'Beata', 1992, 1600)
worker_5 = worker(5, 'Czesław', 1980, 2000)
worker_6 = worker(6, 'Cecylia', 1983, 2100)
worker_7 = worker(7, 'Daniel', 1976, 1900)


for person in worker._registry:
        if person.age >= 30:
            old_group_sum += person.salary
            old_group_count += 1

young_avg = (worker.salary_sum - old_group_sum)/(worker.workers_count - old_group_count)
old_avg = old_group_sum/old_group_count
print(young_avg, old_avg) 
