#!/usr/bin/env python3

# SINGLETON
class SemanticMapping():
    '''
    description: class used to describe the semantic mapping
    '''
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.persons = []
        self.seats = []

class Person():
    '''
    description: class used to describe a person
    '''
    def __init__(self):
        self.reid_id = None
        self.party_id = None
        self.name = None
        self.favorite_drink = None
        self.gender = None
        self.age_range = None
        self.fashion = []
