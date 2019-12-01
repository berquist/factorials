r"""Various (in)correct implementations of the double factorial function:
http://mathworld.wolfram.com/DoubleFactorial.html"""

import math
from functools import reduce
from typing import Union

import numpy as np
import scipy.special as sps
from numpy import pi
from pytest import approx, raises


def dfact0(n):
    """https://stackoverflow.com/a/4740229/"""
    return reduce(int.__mul__, range(n, 0, -2))


def dfact1(x: Union[int, float, complex]) -> Union[float, complex]:
    """https://stackoverflow.com/a/36779406/"""
    n = (x + 1.0) / 2.0
    return 2.0 ** n * sps.gamma(n + 0.5) / (pi ** (0.5))


def dfact2(n: int) -> int:
    """https://stackoverflow.com/a/23252938/"""
    k = int((n + 1) / 2)
    if n % 2 == 1:
        return math.factorial(2 * k) / (2 ** k * math.factorial(k))
    return 2 ** k * math.factorial(k)


def dfact3(n: int) -> int:
    """https://stackoverflow.com/a/54698884/"""
    return math.prod(range(n, 0, -2))


def is_complex(n):
    """Return true if the input is
    - a scalar and complex, or
    - a ndarray and _any_ elements are complex.
    """
    return isinstance(n, complex) or (
        isinstance(n, np.ndarray) and np.iscomplex(n).any()
    )


def dfact4(n, *, zero_dfact_is_one: bool = True):
    if is_complex(n):
        # TODO what is a "positive complex number" or a "negative complex
        # number"?
        return dfact5(n)
    else:
        if n > 0:
            return dfact2(n)
        elif n < 0:
            return dfact1(n)
        elif zero_dfact_is_one:
            return dfact2(n)
        else:
            return dfact1(n)


def dfact5(z):
    """https://math.stackexchange.com/a/2640174/"""
    return (
        (2 ** ((1 + (2 * z) - np.cos(pi * z)) / 4))
        * (pi ** ((np.cos(pi * z) - 1) / 4))
        * sps.gamma((0.5 * z) + 1)
    )


def test_dfact0_real() -> None:
    # This implementation only works for n >= 1.
    with raises(TypeError) as excinfo:
        dfact0(0)
    assert "reduce() of empty sequence with no initial value" in str(excinfo.value)
    assert dfact0(1) == 1
    assert dfact0(2) == 2
    assert dfact0(3) == (3 * 1)
    assert dfact0(4) == (4 * 2)
    assert dfact0(5) == (5 * 3 * 1)
    assert dfact0(6) == (6 * 4 * 2)
    assert dfact0(7) == (7 * 5 * 3 * 1)
    assert dfact0(8) == (8 * 6 * 4 * 2)

    # This implementation doesn't work with negative numbers.
    with raises(TypeError) as excinfo:
        dfact0(-1)
    with raises(TypeError) as excinfo:
        dfact0(-3)
    with raises(TypeError) as excinfo:
        dfact0(-5)


def test_dfact1_real() -> None:
    # This implementation is only valid for positive and negative _odd_
    # numbers
    assert dfact1(0) == approx(0.7978845608028655, 1e-12)
    assert dfact1(1) == approx(1.0, 1e-12)
    assert dfact1(2) == approx(1.595769121605731, 1e-12)
    assert dfact1(3) == approx((3.0 * 1.0), 1e-12)
    assert dfact1(4) == approx(6.383076486422924, 1e-12)
    assert dfact1(5) == approx((5.0 * 3.0 * 1.0), 1e-12)
    assert dfact1(6) == approx(38.29845891853755, 1e-12)
    assert dfact1(7) == approx((7.0 * 5.0 * 3.0 * 1.0), 1e-12)
    assert dfact1(8) == approx(306.3876713483004, 1e-12)

    assert dfact1(-1) == approx(1.0, 1e-12)
    assert dfact1(-3) == approx(-1.0, 1e-12)
    assert dfact1(-5) == approx((1.0 / 3.0), 1e-12)

    assert dfact1(-2) == math.inf
    assert dfact1(-4) == math.inf


def test_dfact2_real() -> None:
    assert dfact2(0) == 1
    assert dfact2(1) == 1
    assert dfact2(2) == 2
    assert dfact2(3) == (3 * 1)
    assert dfact2(4) == (4 * 2)
    assert dfact2(5) == (5 * 3 * 1)
    assert dfact2(6) == (6 * 4 * 2)
    assert dfact2(7) == (7 * 5 * 3 * 1)
    assert dfact2(8) == (8 * 6 * 4 * 2)

    assert dfact2(-1) == 1
    with raises(ValueError) as excinfo:
        dfact2(-3)
        dfact2(-5)
    assert "factorial() not defined for negative values" in str(excinfo.value)


def test_dfact3_real() -> None:
    assert dfact3(0) == 1
    assert dfact3(1) == 1
    assert dfact3(2) == 2
    assert dfact3(3) == (3 * 1)
    assert dfact3(4) == (4 * 2)
    assert dfact3(5) == (5 * 3 * 1)
    assert dfact3(6) == (6 * 4 * 2)
    assert dfact3(7) == (7 * 5 * 3 * 1)
    assert dfact3(8) == (8 * 6 * 4 * 2)

    # This is artificially allowed but the answers are incorrect!
    assert dfact3(-1) == 1
    assert dfact3(-3) == 1
    assert dfact3(-5) == 1

    assert dfact3(-2) == 1
    assert dfact3(-4) == 1


def test_dfact4_real() -> None:
    assert dfact4(0) == 1
    assert dfact4(0, zero_dfact_is_one=False) == approx(0.7978845608028655, 1e-12)
    assert dfact4(1) == 1
    assert dfact4(2) == 2
    assert dfact4(3) == (3 * 1)
    assert dfact4(4) == (4 * 2)
    assert dfact4(5) == (5 * 3 * 1)
    assert dfact4(6) == (6 * 4 * 2)
    assert dfact4(7) == (7 * 5 * 3 * 1)
    assert dfact4(8) == (8 * 6 * 4 * 2)
    assert dfact4(-1) == approx(1.0, 1e-12)
    assert dfact4(-2) == math.inf
    assert dfact4(-3) == approx(-1.0, 1e-12)
    assert dfact4(-4) == math.inf
    assert dfact4(-5) == approx((1.0 / 3.0), 1e-12)


def test_dfact5_real() -> None:
    assert dfact5(0) == 1
    # assert dfact5(0, zero_dfact_is_one=False) == approx(0.7978845608028655, 1e-12)
    assert dfact5(1) == approx(1.0, 1e-12)
    assert dfact5(2) == 2
    assert dfact5(3) == (3 * 1)
    assert dfact5(4) == (4 * 2)
    assert dfact5(5) == approx((5 * 3 * 1), 1e-12)
    assert dfact5(6) == (6 * 4 * 2)
    assert dfact5(7) == (7 * 5 * 3 * 1)
    assert dfact5(8) == (8 * 6 * 4 * 2)
    assert dfact5(-1) == approx(1.0, 1e-12)
    assert dfact5(-2) == math.inf
    assert dfact5(-3) == approx(-1.0, 1e-12)
    assert dfact5(-4) == math.inf
    assert dfact5(-5) == approx((1.0 / 3.0), 1e-12)


def test_dfact1_complex() -> None:
    # These aren't actually correct!
    assert dfact1(2j) == approx(0.3846601192054938 + 0.15879441977887088j)
    assert dfact1(2 + 2j) == approx(0.4517313988532462 + 1.0869090779687303j)


def test_dfact5_complex() -> None:
    assert dfact5(2j) == approx(5777269344856.415 + 2384957752879.5967j)
    assert dfact5(2 + 2j) == approx(6784623183953.537 + 16324454195472.117j)
