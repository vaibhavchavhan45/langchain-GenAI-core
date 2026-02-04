from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'ak'
    age: Optional[int] = None # if age is provided then value will be used, if not then None will be used as age
    email: EmailStr
    cgpa: float = Field(lt=10, gt=0, default='FAIL', description= "A decinmal value representing cgpa of a student")
    
student = Student(name = 'vc', email = "vc@gmail.com", cgpa = 9.2)

print(student)
print(student.age)

student_json = student.model_dump_json() # convert this define schema to json
print(student_json)