"""Compatibility shim: makes the repos' OLD spikingjelly imports
(`spikingjelly.clock_driven.*`, `spikingjelly.cext.neuron`) resolve to the NEW
`spikingjelly.activation_based` API, with every spiking neuron forced to
multi-step + Triton backend (`step_mode='m', backend='triton'`).

Import this module BEFORE importing any repo model code.
Set env SJ_NEURON_BACKEND=torch to A/B-compare against the eager torch backend.
"""
import os, sys, types
import spikingjelly
from spikingjelly.activation_based import (
    neuron as AN, layer as AL, functional as AF, surrogate as AS,
)

BACKEND = os.environ.get("SJ_NEURON_BACKEND", "triton")
print(f"[sj_compat] neuron backend = {BACKEND} (step_mode='m')")


def _sg(s):
    return s if s is not None else AS.Sigmoid()


class MultiStepLIFNode(AN.LIFNode):
    def __init__(self, tau=2.0, decay_input=True, v_threshold=1.0, v_reset=0.0,
                 surrogate_function=None, detach_reset=False, backend=None, **kw):
        super().__init__(tau=tau, decay_input=decay_input, v_threshold=v_threshold,
                         v_reset=v_reset, surrogate_function=_sg(surrogate_function),
                         detach_reset=detach_reset, step_mode="m", backend=BACKEND)


class MultiStepIFNode(AN.IFNode):
    def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function=None,
                 detach_reset=False, backend=None, **kw):
        super().__init__(v_threshold=v_threshold, v_reset=v_reset,
                         surrogate_function=_sg(surrogate_function),
                         detach_reset=detach_reset, step_mode="m", backend=BACKEND)


class MultiStepParametricLIFNode(AN.ParametricLIFNode):
    def __init__(self, init_tau=2.0, decay_input=True, v_threshold=1.0, v_reset=0.0,
                 surrogate_function=None, detach_reset=False, backend=None, **kw):
        super().__init__(init_tau=init_tau, decay_input=decay_input, v_threshold=v_threshold,
                         v_reset=v_reset, surrogate_function=_sg(surrogate_function),
                         detach_reset=detach_reset, step_mode="m", backend=BACKEND)


def _register(fqname, **attrs):
    m = types.ModuleType(fqname)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[fqname] = m
    return m


_neuron = _register("spikingjelly.clock_driven.neuron",
                    MultiStepLIFNode=MultiStepLIFNode,
                    MultiStepIFNode=MultiStepIFNode,
                    MultiStepParametricLIFNode=MultiStepParametricLIFNode,
                    LIFNode=AN.LIFNode, IFNode=AN.IFNode, ParametricLIFNode=AN.ParametricLIFNode)
_layer = _register("spikingjelly.clock_driven.layer", SeqToANNContainer=AL.SeqToANNContainer)
_functional = _register("spikingjelly.clock_driven.functional",
                        reset_net=AF.reset_net, set_step_mode=AF.set_step_mode)
_surrogate = _register("spikingjelly.clock_driven.surrogate",
                       **{n: getattr(AS, n) for n in dir(AS) if not n.startswith("_")})
_cd = _register("spikingjelly.clock_driven",
                neuron=_neuron, layer=_layer, functional=_functional, surrogate=_surrogate)
_cext_neuron = _register("spikingjelly.cext.neuron", MultiStepIFNode=MultiStepIFNode)
_cext = _register("spikingjelly.cext", neuron=_cext_neuron)

spikingjelly.clock_driven = _cd
spikingjelly.cext = _cext
