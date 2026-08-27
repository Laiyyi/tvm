"""OPT-125M -> Relax -> DistIR (TP=4), with the full optimization pipeline.

`TestOPT.py` only ran LegalizeOps before the DistIR passes, which is lowering,
not optimization: every op stayed its own kernel. This file mirrors the phase
structure of the official `optimize_llm.py` tutorial and slots the DistIR
passes in between phase 2 and phase 5.

Measured on OPT-125M / TP=4, after LiftTransformParams + DeadCodeElimination:

    LegalizeOps only          64 PrimFuncs, 660 call_tir in main
    this pipeline             25 PrimFuncs, 271 call_tir in main
    allreduce                 24 in both  <- fusion does not disturb the sharding

Two things are deliberately left out; see PHASE 5 and the dlight comment.
"""

import os
import sys

import tvm
from tvm import relax
from tvm.relax import transform as X

# The IRModule lives next door so this file stays readable; let it run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from opt125m_module import Module  # noqa: E402

T = relax.distributed.transform

NUM_LAYERS = 12
NUM_WORKERS = 4  # must match Module's I.module_attrs({"device_num": ...})

target = tvm.target.Target("llvm")


def optimize(mod):
    """Module -> lowered, sharded, param-lifted IRModule."""
    with target:
        # ---- PHASE 1. High-level operator graph -------------------------------
     
        mod = X.FuseTransposeMatmul()(mod)

        # ---- PHASE 2. Lower to TIR ("zero" pipeline) --------------------------
     
        mod = X.LegalizeOps()(mod)
        mod = X.AnnotateTIROpPattern()(mod)
        mod = X.FoldConstant()(mod)
        mod = X.FuseOps()(mod)
        mod = X.FuseTIR()(mod)

        # ---- PHASE 3. DistIR -------------------------------------------------
        for dist_pass in (
            T.PropagateSharding(),
            T.LowerGlobalViewToLocalView(),
            T.LegalizeRedistribute(),
            T.LowerDistIR(),
        ):
            mod = dist_pass(mod)

        # ---- PHASE 4. Get the weight sharding out of the hot path -------------
   
        mod = X.LiftTransformParams()(mod)
        mod = X.DeadCodeElimination()(mod)

        # dlight is NOT applied. The tutorial uses dlight.gpu.* because its
        # target is cuda; the llvm equivalents are dlight.cpu.GEMV/Reduction, and
        # ApplyDefaultSchedule segfaults on this module (s_tir/schedule/error.cc
        # RenderReport). It segfaults without the DistIR passes too, so it is a
        # dlight.cpu bug, not a DistIR interaction. For an llvm target the LLVM
        # vectorizer covers this anyway. Re-enable once that is fixed:
        #     from tvm.s_tir import dlight
        #     mod = dlight.ApplyDefaultSchedule(dlight.cpu.GEMV(),
        #                                       dlight.cpu.Reduction())(mod)

        # ---- PHASE 5. Lower to VM bytecode -----------------------------------
        # Not spelled out here: tvm.compile's default relax_pipeline already runs
        # exactly these passes (RewriteDataflowReshape ... AttachGlobalSymbol).
        # Writing them out only risks running LegalizeOps twice.
        #
        # `tvm.compile(mod, target=target)` currently dies inside _vmlink with
        #     Cannot find type info for type_index=<differs every run>
        # on any module containing R.ccl.* -- a garbage type_index, i.e. a
        # corrupt object header, not a registration problem. It reproduces on a
        # 25-line 2-worker MLP and is independent of this pipeline (the
        # LegalizeOps-only version dies the same way), so it is left out until
        # that is fixed.
    return mod


def report(mod):
    text = mod["main"].script()
    counts = {
        "PrimFuncs": sum(
            1 for _, f in mod.functions_items() if isinstance(f, tvm.tirx.PrimFunc)
        ),
        "call_tir in main": text.count("R.call_tir("),
        "allreduce": text.count("ccl.allreduce"),
        "scatter in main": text.count("scatter_from_worker0"),
        "broadcast in main": text.count("broadcast_from_worker0"),
    }
    width = max(len(k) for k in counts)
    for key, value in counts.items():
        print(f"  {key:<{width}} : {value}")
    return counts


def main():
    mod = optimize(Module)
    print("relax functions:", sorted(
        gv.name_hint for gv, f in mod.functions_items() if isinstance(f, relax.Function)
    ))
    counts = report(mod)

    # Row-parallel out_proj and fc2 each need one allreduce per layer, and both
    # were inferred rather than annotated -- that inference is the thing under
    # test, so assert on it.
    assert counts["allreduce"] == 2 * NUM_LAYERS, counts
    # After lifting, main must not reshard weights any more. The one surviving
    # broadcast is input_ids, which is a runtime input and has to be broadcast
    # on every call.
    assert counts["scatter in main"] == 0, counts
    assert counts["broadcast in main"] == 1, counts

    q_weight_rows = 768 // NUM_WORKERS
    local = [
        v.name for v in mod["main"].params
        if len(v.ty.shape.values) == 1 and int(v.ty.shape.values[0]) == q_weight_rows
    ]
    assert local, "main should take local shards, not global weights"
    print(f"ok: main takes local shards ({len(local)} params of extent {q_weight_rows})")


if __name__ == "__main__":
    main()
