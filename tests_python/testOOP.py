# Project: python test OOP Classes
# Author: Trinidad Martín Campos
# Created: November 6, 2023
# Description: This script performs a simple test for python classes

from abc import ABC, abstractmethod  # to create abstract classes


class AuthMixin:
    is_authenticated = False

    def login(self, taken_password):
        if self.password == taken_password:
            self.is_authenticated = True
            print(f"{self.username} is authenticated")
        else:
            print("Wrong password!")

    def logout(self):
        self.is_authenticated = False
        print(f"{self.username} is loggouted")


class AbstractAdmin(ABC):
    @abstractmethod
    def login():
        pass

    @abstractmethod
    def logout():
        pass

    @abstractmethod
    def create_content():
        pass

    @abstractmethod
    def update_content():
        pass

    @abstractmethod
    def delete_content():
        pass


class User(AuthMixin):
    role = "User"

    def __init__(self, username, password):
        self.username = username
        self.password = password

    @property
    def password(self):
        return self._password

    @password.setter
    def password(self, new_password):
        if isinstance(new_password, str):
            if len(new_password) >= 8:
                self._password = new_password
            else:
                print("The password length should be >= 8.")
        else:
            print("The password should be string.")

    # ========== magic methods ==========
    def __repr__(self):
        return f"{self.role}: {self.username}"

    def __eq__(self, other):
        # the User instance
        if isinstance(other, User):
            return self.username == other.username

        # string
        if isinstance(other, str):
            return self.username == other

        return False


# ========== end magic methods ==========


class Admin(User, AbstractAdmin):
    role = "Admin"

    def create_content(self):
        print(f"{self.username} creates the content")

    def update_content(self):
        print(f"{self.username} updates the content")

    def delete_content(self):
        print(f"{self.username} deletes the content")


bob = User("bob.user123", "b123b321")
frank = Admin("frank.top.admin", "super_secret_admin_password")

print(bob)
print(frank)

print(bob == frank)
print(bob == "bob.user123")
print(frank == "frank.admin")

