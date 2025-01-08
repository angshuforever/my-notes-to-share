# Python Data Types: Ranges and Examples

| Data Type | Range/Default Values | Code Examples |
|-----------|-------------------|---------------|
| `str` | - Min length: 0 (empty string)<br>- Max length: Limited by memory<br>- Default: "" (empty string) | ```python
# String creation
text = "Hello, World!"
multi = """Multiple
lines"""
empty = ''
escaped = "Tab\tNewline\n"
formatted = f"Value: {42}"
``` |
| `int` | - Min: No theoretical limit<br>- Max: Limited by memory<br>- Default: 0 | ```python
# Integer examples
x = 42
big = 1_000_000
negative = -123
from_str = int("100")
hex_num = 0xFF  # 255
bin_num = 0b1010  # 10
``` |
| `float` | - Min: -1.8e308<br>- Max: 1.8e308<br>- Precision: ~15-17 digits<br>- Default: 0.0 | ```python
# Float examples
pi = 3.14159
sci = 1.23e-4
inf = float('inf')
nan = float('nan')
from_int = float(42)
``` |
| `complex` | - Real & Imaginary parts: same as float<br>- Default: 0j | ```python
# Complex numbers
c1 = 3 + 4j
c2 = complex(3, 4)
mag = abs(c1)  # magnitude
real = c1.real  # 3.0
imag = c1.imag  # 4.0
``` |
| `list` | - Min size: 0 (empty)<br>- Max size: Limited by memory<br>- Default: [] | ```python
# List operations
nums = [1, 2, 3]
mixed = [1, "two", 3.0]
nested = [[1, 2], [3, 4]]
nums.append(4)
first = nums[0]
sliced = nums[1:3]
``` |
| `tuple` | - Min size: 0 (empty)<br>- Max size: Limited by memory<br>- Default: () | ```python
# Tuple operations
coords = (3, 4)
single = (1,)  # note the comma
empty = tuple()
x, y = coords  # unpacking
nested = ((1, 2), (3, 4))
``` |
| `range` | - Min: No theoretical limit<br>- Max: No theoretical limit<br>- Step: Any non-zero int | ```python
# Range examples
r1 = range(5)      # 0 to 4
r2 = range(1, 6)   # 1 to 5
r3 = range(0, 10, 2)  # even numbers
list(r1)  # [0, 1, 2, 3, 4]
``` |
| `dict` | - Min size: 0 (empty)<br>- Max size: Limited by memory<br>- Default: {} | ```python
# Dictionary operations
d = {'a': 1, 'b': 2}
d['c'] = 3
val = d.get('a', 0)
keys = d.keys()
items = d.items()
nested = {'x': {'y': 2}}
``` |
| `set` | - Min size: 0 (empty)<br>- Max size: Limited by memory<br>- Default: set() | ```python
# Set operations
s1 = {1, 2, 3}
s2 = {3, 4, 5}
union = s1 | s2
inter = s1 & s2
s1.add(4)
s1.remove(1)
``` |
| `frozenset` | - Min size: 0 (empty)<br>- Max size: Limited by memory<br>- Default: frozenset() | ```python
# Frozenset operations
fs = frozenset([1, 2, 3])
fs2 = frozenset([3, 4, 5])
union = fs | fs2
inter = fs & fs2
# fs.add(4)  # Error!
``` |
| `bool` | - Only two values: True/False<br>- Default: False | ```python
# Boolean operations
t = True  # 1
f = False  # 0
b1 = bool(1)    # True
b2 = bool("")   # False
b3 = 5 > 3      # True
``` |
| `bytes` | - Min size: 0<br>- Max size: Limited by memory<br>- Value range: 0-255 per byte<br>- Default: b'' | ```python
# Bytes operations
b1 = b'Hello'
b2 = bytes([65, 66, 67])
b3 = b'ABC'
# b3[0] = 68  # Error!
val = b3[0]  # 65
``` |
| `bytearray` | - Min size: 0<br>- Max size: Limited by memory<br>- Value range: 0-255 per byte<br>- Default: bytearray() | ```python
# Bytearray operations
ba = bytearray([65, 66, 67])
ba[0] = 68  # OK
ba.append(69)
ba.extend(b'FGH')
``` |
| `memoryview` | - Size: Based on original object<br>- Default: N/A (requires source) | ```python
# Memoryview operations
b = bytes([65, 66, 67, 68])
mv = memoryview(b)
list(mv)  # [65, 66, 67, 68]
mv[0]  # 65
``` |
| `NoneType` | - Single value: None<br>- Default: None | ```python
# None usage
x = None
if x is None:
    print("undefined")
def no_return():
    pass  # returns None
``` |