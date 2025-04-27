import ctypes
import os

lib = ctypes.CDLL("./build/libfortran_code.so")

lib.add_numbers.argtypes = [ctypes.c_double, ctypes.c_double]
lib.add_numbers.restype = ctypes.c_double

a = 3.5
b = 2.5
result = lib.add_numbers(a, b)

print(f"Résultat de {a} + {b} = {result}")
