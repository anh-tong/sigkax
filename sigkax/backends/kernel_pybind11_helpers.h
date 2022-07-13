/*
The idea of this file is from https://github.com/google/jax/blob/main/jaxlib/kernel_pybind11_helpers.h
It's better to include jaxlib but I don't know how
*/

#ifndef SIGKAX_KERNEL_PYBIND11_HELPERS_H_
#define SIGKAX_KERNEL_PYBIND11_HELPERS_H_


#include <pybind11/pybind11.h>
#include "kernel_helpers.h"

namespace sigkax
{
    // Pack a descriptor to pybind11::bytes
    template <typename T>
    pybind11::bytes PackDescriptor(const T &descriptor)
    {
        return pybind11::bytes(PackDescriptorAsString(descriptor));
    }

    // Encapsulate a C function to Python so that we can use it in Python
    // For example, in Python, JAX uses encapsulated functions in its custom call
    template <typename T>
    pybind11::capsule EncapsulateFunction(T *fn)
    {
        return pybind11::capsule(bit_cast<void *>(fn), "xla._CUSTOM_CALL_TARGET");
    }
}

#endif /* SIGKAX_KERNEL_PYBIND11_HELPERS_H_ */
