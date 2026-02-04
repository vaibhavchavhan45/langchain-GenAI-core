from pydantic import BaseModel

class Student(BaseModel):
    name: str
    age: int

student = Student(name = 'vaibhav', age = 20)

## OR
# new_student = {'name': "vaibhav", 'age' : 20}
# student = Student(**new_student)

## OR
# student = Student(**{'name': "vaibhav", 'age': 20})

## OR
# new_student = dict(name = "vaibhav", age = 20)
# student = Student(**new_student)


print(student)