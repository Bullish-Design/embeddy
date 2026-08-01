import os
import sys

@dataclass
class Person:
    """A person with a name and an age."""

    name: str
    age: int

    def greet(self, greeting="Hi"):
        """Return a greeting for this person."""
        return f"{greeting}, {self.name}"

    @staticmethod
    def from_name(name):
        return Person(name, 0)


def top_level(x):
    if x:
        return x + 1
    return x - 1
