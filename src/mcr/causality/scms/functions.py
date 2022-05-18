import jax.nn
import torch
import jax.numpy as jnp

class StructuralFunction:

    def __init__(self, fnc, inv=None, transform=None, additive=False, binary=False, raw=None):
        self.fnc = fnc
        self.inv_fnc = inv
        self.reverse_transform = transform
        self.additive = additive
        self.binary = binary
        self.raw_fnc = raw

    @staticmethod
    def get_zero(obj):
        if isinstance(obj, torch.Tensor):
            zero = torch.tensor([0.0])
        else:
            zero = jnp.array([0.0])
        return zero

    def __call__(self, *args, **kwargs):
        return self.fnc(*args, **kwargs)

    def raw(self, x_pa, *args, **kwargs):
        if self.raw_fnc is not None:
            return self.raw_fnc(x_pa, *args, **kwargs)
        elif self.additive:
            return self.__call__(x_pa, StructuralFunction.get_zero(x_pa), *args, **kwargs)
        else:
            raise NotImplementedError('raw not implemented for non-additive functions')

    def is_transformable(self):
        return self.reverse_transform is not None

    def transform(self, x_pa, x_j, *args, **kwargs):
        return self.reverse_transform(x_pa, x_j, *args, **kwargs)

    def is_invertible(self):
        return self.inv_fnc is not None or self.additive

    def inv(self, x_pa, x_j, *args, **kwargs):
        if self.inv_fnc is None:
            if self.additive:
                zero = StructuralFunction.get_zero(x_pa)
                x_j_wo = self.fnc(x_pa, zero)
                u_j = x_j - x_j_wo
                return u_j
            else:
                raise RuntimeError('Function is not inv')
        else:
            return self.inv_fnc(x_pa, x_j, *args, **kwargs)

def sigmoid(x_pa):
    input = jnp.sum(x_pa, axis=-1)
    p = jax.nn.sigmoid(input)
    return p

def sigmoidal_binomial(x_pa, u_j):
    p = sigmoid(x_pa)
    output = jnp.greater_equal(p, u_j.flatten()) * 1.0
    return output

sigmoidal_binomial = StructuralFunction(sigmoidal_binomial, binary=True, raw=sigmoid)

def nonlinear_additive(x_pa, u_j, coeffs=None):
    if coeffs is None:
        coeffs = jnp.ones(x_pa.shape[1])
    input = 0
    for jj in range(len(coeffs)):
        input = input + jnp.power(x_pa[:, jj], jj+1)
    output = input.flatten() + u_j.flatten()
    return output

nonlinear_additive = StructuralFunction(nonlinear_additive, additive=True)

def sigmoid_torch(x_pa):
    input = torch.sum(x_pa, dim=-1, keepdim=True)
    return torch.sigmoid(input)

def sigmoidal_binomial_torch(x_pa, u_j):
    input = sigmoid_torch(x_pa)
    output = torch.greater_equal(input, u_j) * 1.0
    return output

sigmoidal_binomial_torch = StructuralFunction(sigmoidal_binomial_torch, binary=True, raw=sigmoid_torch)

def nonlinear_additive_torch(x_pa, u_j, coeffs=None):
    if coeffs is None:
        coeffs = jnp.ones(x_pa.shape[1])
    input = 0
    for jj in range(len(coeffs)):
        input = input + jnp.power(x_pa[:, jj], jj+1)
    output = input + u_j
    return output

nonlinear_additive_torch = StructuralFunction(nonlinear_additive_torch, additive=True)


def linear_additive_torch(x_pa, u_j):
    result = u_j
    if x_pa.shape[1] > 0:
        mean_pars = x_pa.sum(dim=-1, keepdim=True)
        result = mean_pars + result
    return result

linear_additive_torch = StructuralFunction(linear_additive_torch, additive=True)

def linear_additive(x_pa, u_j):
    result = u_j.flatten()
    if x_pa.shape[1] > 0:
        mean_pars = jnp.sum(x_pa, axis=1)
        result = mean_pars.flatten() + result
    return result

linear_additive = StructuralFunction(linear_additive, additive=True)