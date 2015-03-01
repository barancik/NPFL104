#!/usr/bin/python
# -*- coding: utf-8 -*-

import unittest
import string
import pdb
from collections import Counter
import math

LOWER_ASCII=string.ascii_lowercase
WHITESPACE_AND_PUNCTUATION=string.punctuation+string.whitespace

def translate(string):
    """Write a function translate() that will translate a text into "rövarspråket" 
    (Swedish for "robber's language"). That is, double every consonant and place an 
    occurrence of "o" in between. For example, translate("this is fun") should 
    return the string "tothohisos isos fofunon"."""
    return "".join([x if x in "aeiouAEIOU" or x==" " else "{0}o{0}".format(x) for x in string ])

def reverse(string):
    """Define a function reverse() that computes the reversal of a string. 
    For example, reverse("I am testing") should return the string "gnitset ma I"."""
    return string[::-1]

def is_palindrome(string):
    """Define a function is_palindrome() that recognizes palindromes (i.e. words 
    that look the same written backwards). For example, is_palindrome("radar") 
    should return True."""
    return string == reverse(string)

def is_pangram(string):
    """A pangram is a sentence that contains all the letters of the English alphabet 
    at least once, for example: The quick brown fox jumps over the lazy dog. Your task 
    here is to write a function to check a sentence to see if it is a pangram or not."""
    string=string.translate(None,WHITESPACE_AND_PUNCTUATION).lower()
    return Counter(LOWER_ASCII).keys() == Counter(string).keys()

def char_freq(string):
    """ Write a function char_freq() that takes a string and builds a frequency listing 
    of the characters contained in it. Represent the frequency listing as a Python 
    dictionary. Try it with something like char_freq("abbabcbdbabdbdbabababcbcbab")."""
    return Counter(string)

def is_member(x,array):
    """Write a function is_member() that takes a value (i.e. a number, string, etc) 
    x and a list of values a, and returns True if x is a member of a, False otherwise. 
    (Note that this is exactly what the in operator does, but for the sake of the
    exercise you should pretend Python did not have this operator.)"""
    for member in array:
        if x == member:
             return True
    return False

def generate_n_chars(n,char):
    """Define a function generate_n_chars() that takes an integer n and a character 
    c and returns a string, n characters long, consisting only of c:s. For example, 
    generate_n_chars(5,"x") should return the string "xxxxx". (Python is unusual in 
    that you can actually write an expression 5 * "x" that will evaluate to "xxxxx". 
    For the sake of the exercise you should ignore that the problem can be solved in 
    this manner.)"""
    try:
       n=int(n)
       char=str(char)
    except:
       raise TypeError("Integer and a char/string needed.")
    return "".join(char for x in range(n))

def max_in_list(arr):
    """The function max() from exercise 1) and the function max_of_three() from 
    exercise 2) will only work for two and three numbers, respectively. But suppose 
    we have a much larger number of numbers, or suppose we cannot tell in advance 
    how many they are? Write a function max_in_list() that takes a list of numbers 
    and returns the largest one."""
    if not arr:
        raise ValueError("Empty list")
    maximum=arr[0]
    for x in arr[1:]:
        maximum=max(maximum,x)
    return maximum

def palindrome(stringe):
    """Write a version of a palindrome recognizer that also accepts phrase 
    palindromes such as "Go hang a salami I'm a lasagna hog.", "Was it a rat 
    I saw?", "Step on no pets", "Sit on a potato pan, Otis", "Lisa Bonet ate 
    no basil", "Satan, oscillate my metallic sonatas", "I roamed under it as 
    a tired nude Maori", "Rise to vote sir", or the exclamation "Dammit, I'm 
    mad!". Note that punctuation, capitalization, and spacing are usually 
    ignored."""
    string=string.translate(None,WHITESPACE_AND_PUNCTUATION).lower()
    return string == string[::-1]

def double_char(str):
    """Given a string, return a string where for every char in the original, 
    there are two chars.
    double_char('The') → 'TThhee'
    double_char('AAbb') → 'AAAAbbbb'
    double_char('Hi-There') → 'HHii--TThheerree' """
    return "".join([x+x for x in str])

class Circle():
    """ Simple class for a circle. It is determined by a coordinant of
        its centre (i.e. (0,0)) and its radius"""
    def __init__(self,centre,radius):  
        self.centre=centre
        self.radius=radius
  
    def area(self):
        return self.radius ** 2 * math.pi
   
    def __contains__(self,point):
        return math.sqrt(((self.centre[0]-point[0]) ** 2 + (self.centre[1]-point[1]) ** 2)) <= self.radius

#pdb.set_trace()

class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
         self.string = "mackerel"
         self.circle = Circle((0,0),5)
    
    def test_translate(self):
        self.assertEqual(translate("this is fun"),"tothohisos isos fofunon")

    def test_translate2(self):
        self.assertNotEqual(translate("this is fun"),self.string)
    
    def test_translate3(self):
        self.assertEqual(translate("A e"),"A e")
      
    def test_is_pangram(self):
        self.assertTrue(is_pangram("The quick brown fox jumps over the lazy dog."))

    def test_is_pangram2(self):
        self.assertFalse(is_pangram(self.string))

    def test_char_freq(self):
        self.assertNotEqual(char_freq(self.string),{'e': 2, 'a': 1, 'c': 1, 'k': 1})

    def test_char_freq2(self):
        self.assertEqual(char_freq("abbabcbdbabdbdbabababcbcbab"),{'b': 14, 'a': 7, 'c': 3, 'd': 3})
 
    def test_reverse(self):
        self.assertEqual(reverse(self.string),"lerekcam")

    def test_reverse2(self):
        self.assertNotEqual(reverse(self.string),self.string)
 
    def test_is_palindrome(self):
        self.assertTrue(is_palindrome("saippuakalasalakauppias"))
 
    def test_is_palindrome2(self):
        self.assertFalse(is_palindrome(self.string))

    def test_is_member(self):
        self.assertTrue(is_member(self.string,[6,56,"R",self.string,"broccoli"]))

    def test_is_member2(self):
        self.assertFalse(is_member(self.string,[6,56,"R","obese dragonfish","broccoli"]))

    def test_is_member3(self):
        self.assertFalse(is_member(self.string,[]))
 
    def test_generate_n_chars(self):
        self.assertEqual(generate_n_chars(5,"c"),"ccccc")

    def test_generate_n_chars2(self):
        self.assertRaisesRegexp(TypeError,"Integer and a char/string needed.",generate_n_chars,"c","T")
    
    def test_max_in_list(self):
        self.assertEqual(max_in_list([3,4,9,-5,6]),9)

    def test_max_in_list2(self):
        self.assertRaisesRegexp(ValueError,"Empty list",max_in_list,[])

    def test_palindrome(self):
        self.assertTrue("Go hang a salami I'm a lasagna hog.")
    
    def test_palindrome2(self):
        self.assertTrue("Satan, oscillate my metallic sonatas")

    def test_double_char(self):
        self.assertEqual(double_char(''),'')

    def test_double_char2(self):
        self.assertEqual(double_char('Hi-There'),'HHii--TThheerree')

    def test_double_char3(self):
        self.assertNotEqual(double_char('Hi'),'Hii')
    
    def test_circle1(self):
        self.assertTrue((1,1) in self.circle)

    def test_circle2(self):
        self.assertFalse((5,1) in self.circle)

    def test_circle3(self):
        self.assertEqual(self.circle.area(),78.53981633974483)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)

