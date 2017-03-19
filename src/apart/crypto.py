'''
This module provides an emulated implementation of the Elgamal scheme, along with its homomohpic operations.

The implementation is *emulated*, in the sense that plaintexts are not actually
encrypted. See the description of class :class:`.Ctxt`. However, the product of
public keys, a mechanism used in the protocol, is reflected in the
implementation.

It also provides an emulation of the :func:`.SHA3_hash` hash function.

This module defines the :exc:`.DecryptionError` and :exc:`.ReencryptionError` exception.
'''
import copy
import math
import random

from apart.core.protocol_constants import GROUP_G, GROUP_P


class Ctxt(tuple):
    """Class emulating a ciphertext of the Elgamal scheme.
    
    Ciphertexts and cryptographic operations are not actually implemented. Here,
    a ciphertext is represented by a 2-tuple, containing the plaintext (in
    clear) in the first element, and the (product of) keys in the second
    element. This second element allows to check that e.g. decryption is done
    with the righteous public key (thus avoiding the code of the protocol to
    "cheat" and decrypt ciphertext it should not).
    
    The module constants GROUP_P and GROUP_G of the
    :mod:`~apart.core.protocol_constants` are used for group operations.
    However, it is not a schnorr group that is used, but simply the group Zp*
    """
    def __new__(self, *args):
        return super().__new__(self, args)
    
    def __deepcopy__(self, memo):
        return Ctxt(copy.deepcopy(self[0], memo), copy.deepcopy(self[1], memo))

def Elgamal_keygen(secparam):
    """Emulates the key generation of the Elgamal scheme.
    
    Args:
        secparam (int): this argument is there for compatibility, but is not used.
    
    Returns:
        (int, int): a key pair. Here, (pk, sk) = (r, r), for r a random group element.
    """
    r = random.randint(1, GROUP_P-1)
    return (r, r)

def Elgamal_enc(pk, m):
    """Emulates the encryption operation of the Elgamal scheme.
    
    Args:
        pk (int): the public key to use
        m (*): the plaintext to encrypt (can be type of data)
    
    Returns:
        :obj:`.Ctxt`: an (emulated) ciphertext ``Ctxt(m, pk)``
    """
    return Ctxt(m, pk)

def Elgamal_dec(sk, c):
    """Emulates the decryption operation of the Elgamal scheme.
    
    Checks whether the second element of ``c`` is equal to ``sk``. If so,
    outputs the first element of ``c`` (i.e. the plaintext). Otherwise, raises a
    :exc:`.DecryptionError`.
    
    Args:
        sk (int): the secret key to use in the decryption operation
        c (:obj:`.Ctxt`): the ciphertext to decrypt
    
    Returns:
        any: the plaintext, that can be of any type
    
    Raises:
        :exc:`.DecryptionError` if the ciphertext ``c`` is not encrypted under ``sk``
    
    """  
    if c[1] == sk:
        return c[0]
    else:
        raise DecryptionError("Attempt to decrypt with the wrong secret key: {} provided, while {} expected".format(sk, c[1]))

def Elgamal_ctxt_mult(c1, c2):
    """Emulates the homomorphic multiplication of the Elgamal scheme.
    
    Performs an assertion check, to ensure that ``c1`` and ``c2`` are encrypted
    under the same (product of) key(s). If this verification passes, the group
    multiplication modulo GROUP_P of the fist element of both ciphertexts (i;e.
    their plaintext) is manually performed, and a new ciphertext is constructed
    and returned.
    
    Args:
        c1 (:obj:`.Ctxt`): the first ciphertext, encrypting a group element
        c2 (:obj:`.Ctxt`): the second ciphertext, also encrypting a group element
        
    Returns:
        :obj:`.Ctxt`: the ciphertext encrypting the product of ``c1`` and ``c2``'s plaintexts  
    
    Raises:
        AssertionError: if ``c1`` and ``c2`` are not encrypted under the same (product of) key(s)
    """
    assert c1[1] == c2[1], "Impossible to multiply two ciphertexts not encrypted under the same public key(s)"
    return Ctxt(c1[0]*c2[0] % GROUP_P, c1[1])

def Elgamal_ctxt_div(c1, c2):
    """Emulates the homomorphic division of the Elgamal scheme.
    
    This function is similar to :func:`.Elgamal_ctxt_mult`, except that the
    plaintext of ``c1`` is multiplied by the group inverse of ``c2``'s plaintext.
    
    Args:
        c1 (:obj:`.Ctxt`): the first ciphertext, encrypting a group element
        c2 (:obj:`.Ctxt`): the second ciphertext, encrypting a group element
    
    Returns:
        :obj:`.Ctxt`: the ciphertext encrypting the product of ``c1`` and ``c2``'s plaintexts  
    
    Raises:
        AssertionError: if ``c1`` and ``c2`` are not encrypted under the same (product of) key(s)
    """
    assert c1[1] == c2[1], "Impossible to divide two ciphertexts not encrypted under the same public key(s): ciphertexts have keys {} != {}".format(c1[1] % GROUP_P, c2[1] % GROUP_P)
    return Ctxt(c1[0]*group_inverse(c2[0]) % GROUP_P, c1[1])

def Elgamal_plain_mult(c, m):
    """Emulates the homomorphic plaintext multiplication of the Elgamal scheme.
    
    This function performs the group multiplication modulo GROUP_P of the fist
    element of ``c`` with ``m`` creates a new ciphertext for the result (under
    the same key as ``c``), and returns it.
    
    Args:
        c (:obj:`.Ctxt`): a ciphertext, encrypting a group element
        m (int): a group element
    
    Returns:
        :obj:`.Ctxt`: the ciphertext encrypting the product of ``c``'s plaintext and ``m``  
    """
    
    return Ctxt(c[0]* m % GROUP_P, c[1])

def Elgamal_scalar_exp(c, e):
    """Emulates the homomorphic scalar exponentiation of the Elgamal scheme.
    
    This function performs the modular exponentiation modulo GROUP_P of the first
    element of ``c`` with ``e``, creates a new ciphertext for the result (under
    the same key as ``c``), and returns it.
    
    Args:
        c (:obj:`.Ctxt`): a ciphertext, encrypting a group element
        e (int): any integer
    
    Returns:
        :obj:`.Ctxt`: the ciphertext encrypting the exponentiation of ``c``'s plaintext by ``e``  
    """
    return Ctxt(pow(c[0], e, GROUP_P), c[1])

def Elgamal_key_mult(sk, c):
    """Emulates the homomorphic key multiplication of the Elgamal scheme.
    
    This function performs the group multiplication modulo GROUP_P of the second
    element of ``c1`` with ``sk``, creates a ciphertext with the same plaintext
    as ``c``, but for the new key, and returns it.
    
    Args:
        sk (int): a group element 
        c (:obj:`.Ctxt`): a ciphertext, encrypting any kind of data
        
    
    Returns:
        :obj:`.Ctxt`: the ciphertext encrypting the same plaintext as of ``c``'s, 
        but under a ``sk*sk'`` (where ``sk'`` is the original key under which ``c`` was encrypted)  
    """
    return Ctxt(c[0], c[1]*sk % GROUP_P)

def Elgamal_key_div(sk, c):
    """Emulates the homomorphic key multiplication of the Elgamal scheme.
    
    This function is the inverse of :func:`.Elgamal_key_mult`, and functions in
    the same way. Note that no check is performed: the modular division of the
    keys is performed, even if ``sk`` is not a factor of ``sk'``, the key under
    which ``c`` is encrypted.
    
    Args:
        sk (int): a group element 
        c (:obj:`.Ctxt`): a ciphertext, encrypting any kind of data
        
    
    Returns:
        :obj:`.Ctxt`: the ciphertext encrypting the same plaintext as of ``c``'s, 
        but under a ``sk'/sk`` (where ``sk'`` is the original key under which ``c`` was encrypted)  
    """
    new_sk = c[1]*group_inverse(sk) % GROUP_P
#     if not new_sk.is_integer():
#         raise DecryptionError("Attempt to partially decrypt with an invalid secret: expected factor of {}, got {}".format(c[1], sk))
    return Ctxt(c[0], int(new_sk))

def Elgamal_enc_nopk(cone, m):
    """Emulates the encryption without public key in the Elgamal scheme.
    
    This function performs simply returns a ciphertexts contructed with ``m`` in
    first element, and the same second element as ``cone`` Args:. In the
    considered representation of ciphertexts (see :class:`.Ctxt`, this yields a
    a ciphertext for ``m`` encrypted under the same key as ``cone``).
    
    Note: before the encryption, ``cone`` is re-encrypted using :func:`.Reenc_one`.
    
    Args:
        cone (:obj:`.Ctxt`): the encryptions of one, to perform the encryption
        m (any): the plaintext to encrypt (can be any data)
    
    Returns:
        :obj:`.Ctxt`: the ciphertext encrypting ``m`` under the same key as ``cone``  
    """
    cone = Reenc_one(cone)
    return Ctxt(m, cone[1])

# The two following operations are actually not directly implementable with
# Elgamal: there is no actual homomorphic operations that could replace it
def Elgamal_accumulator_add(caccumulator, v):
    """This function abusively implements the homomorphic addition of an element in an Elgamal encrypted accumulator
    
    This function works on plaintexts that are strings, and performs string
    concatenation to add value ``v`` into the accumulator (a string) encrypted
    into the ciphertext ``caccumulator``. This is *cheating*, since, in an
    actual Elgamal implementation, this homomorphic operation would not be
    possible, at least not in this form.(it would require several ciphertexts
    actually...).
    
    This function goes along with the function :func:`.Elgamal_accumulator_check`.
    
    Args:
        caccumulator (:obj:`.Ctxt`): the ciphertext encrypting an accumulator, respresented here by a string
        v (string): the value to homomorphically add to the encrypted accumulator (by string concatenation)
    
    Returns:
        :obj:`.Ctxt`: the encrypted accumulator updated with the new value ``v`` 
    """
    return Ctxt(caccumulator[0]+v, caccumulator[1])
    
def Elgamal_accumulator_check(caccumulator, v):
    """This function abusively implements the homomorphic membership test on an Elgamal encrypted accumulator
    
    This function works on plaintexts that are strings, and performs string
    string lookups to check if ``v`` is contained into the accumulator (a
    string) encrypted into the ciphertext ``caccumulator``. This is *cheating*,
    since, in an actual Elgamal implementation, this homomorphic operation would
    not be possible, at least not in this form.(it would require several
    ciphertexts actually...). The result is an encrypted boolean.
    
    This function goes along with the function :func:`.Elgamal_accumulator_add`.
    
    Args:
        caccumulator (:obj:`.Ctxt`): the ciphertext encrypting an accumulator, respresented here by a string
        v (string): the value which presence must be tested in the encrypted accumulator (by string lookup)
    
    Returns:
        :obj:`.Ctxt`: an encrypted boolean, equal to True if the value ``v`` is in the encrypted accumulator 
    """
    return Ctxt(v in caccumulator[0], caccumulator[1])

def Reenc_pk(pk, c):
    """Emulates the re-encryption **with** public key in the Elgamal scheme.
    
    This function checks if ``c`` is indeed encrypted under ``pk``, and simply
    returns ``c`` unmodified. In the considered representation of ciphertexts
    (see :class:`.Ctxt`), this is what re-encryption boils down to.
    
    Args:
        pk (int): a public key (a group element)
        c (:obj:`.Ctxt`): the ciphertext to re-encrypt
    
    Returns:
        :obj:`.Ctxt`: a re-encryption of ``c``
        
    Raises:
        :exc:`.ReencryptionError`: if ``c`` is not encrypted under ``pk``
    """
    if c[1] != pk:
        raise ReencryptionError("Attempt to re-encrypt a ciphertext with the wrong public key: expected {} ,received {}".format(c[1], pk))
    return c

def Reenc_nopk(cone, c):
    """Emulates the re-encryption **without** public key in the Elgamal scheme.
    
    This function checks if ``c`` and ``cone`` are indeed encrypted under the
    same public key, then re-encrypts ``cone`` by calling :func:`.Reenc_one`,
    and simply returns ``c`` unmodified. In the considered representation of
    ciphertexts (see :class:`.Ctxt`), this is what re-encryption boils down to.
    
    Args:
        cone (:obj:`.Ctxt`): an encryption of one, to perform the re-encryption
        c (:obj:`.Ctxt`): the ciphertext to re-encrypt
    
    Returns:
        :obj:`.Ctxt`: a re-encryption of ``c``
        
    Raises:
        :exc:`.ReencryptionError`: if ``c`` and ``cone`` are not encrypted under the same public key
    """
    if cone[1] != c[1]:
        raise ReencryptionError("Attempt to re-encrypt a ciphertext with a cone not encrypted under the same key: expected {}, received {}".format(c[1], cone[1]))
    cone = Reenc_one(cone)
    return c

def Reenc_one(cone):
    """Emulates the re-encryption of an encryption of one in the Elgamal scheme.
    
    This function simply returns ``cone`` unmodified. In the considered representation of
    ciphertexts (see :class:`.Ctxt`), this is what re-encryption boils down to.
    
    Args:
        cone (:obj:`.Ctxt`): an encryption of one to reencrypt
    
    Returns:
        :obj:`.Ctxt`: a re-encryption of ``cone``
        
    """
    
    return cone

def SHA3_hash(m):
    """Emulates the SHA3 hash function.
    
    Here, the message ``m`` to be hashed is simply returned unmodified.
    
    Args:
        m (*): the message to hash
    
    Returns:
        bitstring: the hashed message ``m``
    """
    return m

def group_inverse(e):
    """Computes a group inverse modulo GROUP_P (constant defined in :mod:`~apart.core.protocol_constants`)"""
    
    return pow(e, GROUP_P-2, GROUP_P)

class DecryptionError(Exception):
    """Raised when a decryption error occurs, e.g. when trying to decrypt with the wrong secret key."""
    pass

class ReencryptionError(Exception):
    """Raised when a re-encryption error occurs, e.g. when trying to re-encrypt with the inadequate public key or encryption of one"""
    pass

if __name__ == '__main__':
    # Some very basic unit tests
    
    (pk1, sk1) = Elgamal_keygen(128)
    (pk2, sk2) = Elgamal_keygen(128)
    
    print("Trying basic enc/dec")
    c1 = Elgamal_enc(pk1, 42)
    assert Elgamal_dec(sk1, c1) == 42
    try:
        Elgamal_dec(sk2, c1)
        assert False
    except DecryptionError:
        pass
    print('OK')
    
    print()
    print("Trying Keymult")
    c1 = Elgamal_enc(pk1, 42)
    c = Elgamal_key_mult(sk2, c1)
    print(c, " - ", pk1, sk2)
    assert Elgamal_dec(sk1, Elgamal_key_div(sk2, c)) == 42
    assert Elgamal_dec(sk2, Elgamal_key_div(sk1, c)) == 42
    print("OK")
    
    print()
    print("Trying ctxt mult")
    c1 = Elgamal_enc(pk1, 12)
    c2 = Elgamal_enc(pk1, 5)
    assert Elgamal_dec(sk1, Elgamal_ctxt_mult(c1, c2)) == 60
    
    c1 = Elgamal_enc(pk1, math.pow(GROUP_G, 10))
    c2 = Elgamal_enc(pk1, math.pow(GROUP_G, 2))
    assert Elgamal_dec(sk1, Elgamal_ctxt_mult(c1, c2)) == math.pow(GROUP_G, 12)
    print("OK")
    
    print()
    print("Trying ctxt div")
    c1 = Elgamal_enc(pk1, pow(GROUP_G, 10, GROUP_P))
    c2 = Elgamal_enc(pk1, pow(GROUP_G, 2, GROUP_P))
    assert Elgamal_dec(sk1, Elgamal_ctxt_div(c1, c2)) == pow(GROUP_G, 8)
    print("OK")
    
    
    print()
    print("Trying encrypted pk and sk")
    (pk_tmp_1, sk_tmp_1) = Elgamal_keygen(128)
    (pk_tmp_2, sk_tmp_2) = Elgamal_keygen(128)
    (pk_tmp_3, sk_tmp_3) = Elgamal_keygen(128)
    c1 = Elgamal_enc(pk1, pk_tmp_1)
    c2 = Elgamal_plain_mult(c1, pk_tmp_2)
    c3 = Elgamal_plain_mult(c2, pk_tmp_3)
    pk_tmp = Elgamal_dec(sk1, c3)
    c3 = Elgamal_enc(pk_tmp, "test")
    c2 = Elgamal_key_div(sk_tmp_3, c3)
    c1 = Elgamal_key_div(sk_tmp_2, c2)
    m = Elgamal_dec(sk_tmp_1, c1)
    
    assert m == "test"
    print("OK")