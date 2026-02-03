from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int

Person1: Person = {
    'name': 'vaibhav',
    'age' : 22
}
    
print(Person1)


