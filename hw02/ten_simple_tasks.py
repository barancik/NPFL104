#!/usr/bin/python
# -*- coding: utf-8 -*-

def vowel(char):
    """Write a function that takes a character (i.e. a string of length 1) 
    and returns True if it is a vowel, False otherwise."""
    return char in "aeiouAEIOU"


def translate(string):
    """Write a function translate() that will translate a text into "rövarspråket" 
    (Swedish for "robber's language"). That is, double every consonant and place an 
    occurrence of "o" in between. For example, translate("this is fun") should 
    return the string "tothohisos isos fofunon"."""
    return "".join([x if vowel(x) or x==" " else "{0}o{0}".format([x]) for x in string ])


"""
Define a function sum() and a function multiply() that sums and multiplies (respectively) all the numbers in a list of numbers. For example, sum([1, 2, 3, 4]) should return 10, and multiply([1, 2, 3, 4]) should return 24.
"""
def sum(arr):
    return reduce(lambda x,y:x+y,arr,0)

def multiply(arr):
    if not arr:
        return 0
    return reduce(lambda x,y:x*y,arr)

def reverse(string):
    """Define a function reverse() that computes the reversal of a string. 
    For example, reverse("I am testing") should return the string "gnitset ma I"."""
    return string[::-1]

def is_palindrome(string):
    """Define a function is_palindrome() that recognizes palindromes (i.e. words 
    that look the same written backwards). For example, is_palindrome("radar") 
    should return True."""
    return string == reverse(string)

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
    return "".join(char for x in range(n))

def histogram(arr):
    """ Define a procedure histogram() that takes a list of integers and prints a 
    histogram to the screen. For example, histogram([4, 9, 7]) should print the 
    following:
    ****
    *********
    *******
    """
    for num in arr:
        print num*"*"

def max_in_list(arr):
    """The function max() from exercise 1) and the function max_of_three() from 
    exercise 2) will only work for two and three numbers, respectively. But suppose 
    we have a much larger number of numbers, or suppose we cannot tell in advance 
    how many they are? Write a function max_in_list() that takes a list of numbers and 
    returns the largest one."""
    if not arr:
        raise ValueError("Empty list")
    maximum=arr[0]
    for x in arr[1:]:
        maximum=max(maximum,x)
    return maximum

def palindrome(string):
    """Write a version of a palindrome recognizer that also accepts phrase 
    palindromes such as "Go hang a salami I'm a lasagna hog.", "Was it a rat 
    I saw?", "Step on no pets", "Sit on a potato pan, Otis", "Lisa Bonet ate 
    no basil", "Satan, oscillate my metallic sonatas", "I roamed under it as 
    a tired nude Maori", "Rise to vote sir", or the exclamation "Dammit, I'm 
    mad!". Note that punctuation, capitalization, and spacing are usually 
    ignored."""
    from string import punctuation, whitespace
    string=string.translate(None,punctuation+whitespace).lower()
    return string == string[::-1]



def double_char(str):
    """Given a string, return a string where for every char in the original, 
    there are two chars.
    double_char('The') → 'TThhee'
    double_char('AAbb') → 'AAAAbbbb'
    double_char('Hi-There') → 'HHii--TThheerree' """
    return "".join([x+x for x in str])

#import re
def count_code(str):
#    return len(re.findall("co.e",str))
    return  len([st[i:i+4] for i in range(len(st)-3) if st[i:i+2] == "co" if st[i+3] == "e"])
