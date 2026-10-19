/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tvm/relax/transform/meta_schedule.cc
 * \brief Pass for meta_schedule tuning
 */
#include <tvm/ffi/cast.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/relax/transform.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/meta_schedule/database.h>
#include <tvm/tirx/transform.h>

#include "../../s_tir/meta_schedule/module_equality.h"
#include "../../s_tir/meta_schedule/trace_apply.h"

namespace tvm {
namespace relax {
namespace transform {

class MetaScheduleTuner {
 public:
  explicit MetaScheduleTuner(Target target, ffi::String work_dir, int64_t max_trials_global,
                             int64_t max_trials_per_task,
                             ffi::Optional<ffi::Array<ffi::String>> op_names,
                             ffi::Map<ffi::String, runtime::Tensor> params = {})
      : target_(target),
        work_dir_(work_dir),
        max_trials_global_(max_trials_global),
        max_trials_per_task_(max_trials_per_task),
        op_names_(op_names),
        params_(params) {
    normalize_mod_func_ =
        tvm::ffi::Function::GetGlobalRequired("tvm.s_tir.meta_schedule.normalize_mod");
  }

  IRModule TuneIRMod(IRModule mod, transform::PassContext ctx) {
    static ffi::Function tune_relax_func =
        tvm::ffi::Function::GetGlobalRequired("tvm.s_tir.meta_schedule.tune_relax");
    tune_relax_func(mod, params_, target_, work_dir_, max_trials_global_, max_trials_per_task_,
                    op_names_);
    return mod;
  }

  tirx::PrimFunc TuneTIR(tirx::PrimFunc f, transform::PassContext ctx) {
    static ffi::Function tune_tir_func =
        tvm::ffi::Function::GetGlobalRequired("tvm.s_tir.meta_schedule.tune_tir");
    tune_tir_func(normalize_mod_func_(f), target_, work_dir_, max_trials_global_);
    return f;
  }

 private:
  Target target_;
  ffi::String work_dir_;
  int64_t max_trials_global_;
  int64_t max_trials_per_task_;
  ffi::Optional<ffi::Array<ffi::String>> op_names_;
  ffi::Map<ffi::String, runtime::Tensor> params_;
  tvm::ffi::Function normalize_mod_func_;
};

Pass MetaScheduleApplyDatabase(ffi::Optional<ffi::String> work_dir, ffi::Optional<IRModule> dyn_mod, bool enable_warning = false) {
  using tvm::s_tir::meta_schedule::Database;
  Target target = Target::Current(false);
  const std::optional<tvm::ffi::Function> normalize_mod_func_ =
      tvm::ffi::Function::GetGlobalRequired("tvm.s_tir.meta_schedule.normalize_mod");
  TVM_FFI_ICHECK(normalize_mod_func_.has_value()) << "Normalization function is not found.";

  auto pass_func = [=](IRModule mod, PassContext ctx) {
    Database database{ffi::UnsafeInit()};
    if (Database::Current().has_value()) {
      database = Database::Current().value();
    } else {
      TVM_FFI_ICHECK(work_dir.has_value());
      std::filesystem::create_directories(work_dir.value().c_str());
      ffi::String path_workload = work_dir.value() + "/database_workload.json";
      ffi::String path_tuning_record = work_dir.value() + "/database_tuning_record.json";
      LOG(WARNING) << "Creating JSONDatabase. Workload at: " << path_workload
                   << ", Tuning records at: " << path_tuning_record;
      database =
          s_tir::meta_schedule::Database::JSONDatabase(path_workload, path_tuning_record, true);
    }
    /*
     * dyn_mod given: `mod` carries the static shapes the database was tuned on,
     * dyn_mod the symbolic ones we actually compile.  The record is looked up
     * with the static twin, because QueryTuningRecord hashes the module and
     * every ModuleEquality compares shapes, so a symbolic function never finds
     * a statically tuned record.  The trace is then replayed on the symbolic
     * function, which is what we want scheduled.
     *
     * Pairing is by name: the two modules run through the same pass pipeline,
     * but lowering a symbolic shape can emit functions a static one does not,
     * so the loop walks dyn_mod to keep the output complete and treats a
     * missing static twin as "no record".
     */
    IRModule query_mod = mod;
    IRModule target_mod = dyn_mod.value_or(mod);
    ffi::Map<GlobalVar, BaseFunc> result;
    auto mod_eq_structural = s_tir::meta_schedule::ModuleEquality::Create("ignore-tensor");
    for (const auto& iter : target_mod->functions) {
      GlobalVar gv = iter.first;
      BaseFunc base_func = iter.second;
      if (const auto* prim_func_node = base_func.as<tirx::PrimFuncNode>()) {
        tirx::PrimFunc prim_func = ffi::GetRef<tirx::PrimFunc>(prim_func_node);
        IRModule apply_mod = (*normalize_mod_func_)(prim_func).cast<IRModule>();

        IRModule lookup_mod = apply_mod;
        if (dyn_mod.defined()) {
          auto twin = query_mod->global_var_map_.find(gv->name_hint);
          if (twin == query_mod->global_var_map_.end()) {
            result.Set(gv, base_func);
            continue;
          }
          BaseFunc static_func = query_mod->functions[(*twin).second];
          if (const auto* static_node = static_func.as<tirx::PrimFuncNode>()) {
            lookup_mod = (*normalize_mod_func_)(ffi::GetRef<tirx::PrimFunc>(static_node))
                             .cast<IRModule>();
          }
        }

        if (ffi::Optional<s_tir::meta_schedule::TuningRecord> opt_record =
                database->QueryTuningRecord(lookup_mod, target, gv->name_hint)) {
          s_tir::meta_schedule::TuningRecord record = opt_record.value();
          s_tir::Schedule sch = s_tir::Schedule::Traced(
              apply_mod, /*seed=*/-1, /*debug_mask=*/0,
              /*error_render_level=*/s_tir::ScheduleErrorRenderLevel::kDetail);
          bool scheduled = true;
          try {
            if (!mod_eq_structural->Equal(lookup_mod, record->workload->mod)) {
              s_tir::meta_schedule::ScheduleUsingAnchorTrace(sch, record->trace, target);
            } else {
              record->trace->ApplyToSchedule(sch, /*remove_postproc=*/false);
            }
          } catch (const std::exception& error) {
            /*
             * A trace sampled against a constant loop extent can be invalid for
             * a symbolic one -- attention is the usual case, its extents follow
             * the KV length. Keeping the unscheduled function is correct, just
             * slower, so one untransferable kernel does not fail the build.
             */
            scheduled = false;
            if (enable_warning) {
              LOG(WARNING) << "Trace does not transfer to " << gv->name_hint << ": "
                           << error.what();
            }
          }

          if (scheduled) {
            IRModule new_mod = sch->mod();
            TVM_FFI_ICHECK_EQ(new_mod->functions.size(), 1);
            BaseFunc new_base_func = (*new_mod->functions.begin()).second;
            TVM_FFI_ICHECK(new_base_func->IsInstance<tirx::PrimFuncNode>());
            tirx::PrimFunc tuned_prim_func = new_base_func.as_or_throw<tirx::PrimFunc>();
            tirx::PrimFunc new_prim_func = tirx::PrimFunc(/*params=*/tuned_prim_func->params,
                                                          /*body=*/tuned_prim_func->body,
                                                          /*ret_type=*/tuned_prim_func->ret_type,
                                                          /*attrs=*/prim_func->attrs);
            new_prim_func = WithAttr(std::move(new_prim_func), tirx::attr::kIsScheduled, true);
            result.Set(gv, new_prim_func);
            continue;
          }
        } else if (enable_warning) {
          LOG(WARNING) << "Tuning record is not found for primfunc: " << gv->name_hint;
        }
      }
      result.Set(gv, base_func);
    }
    mod = target_mod;
    return IRModule(result,       // functions
                    {},           // map
                    mod->attrs);  // attrs);
  };
  return CreateModulePass(pass_func, 0, "MetaScheduleApplyDatabase", {});
}

Pass MetaScheduleTuneIRMod(ffi::Map<ffi::String, runtime::Tensor> params, ffi::String work_dir,
                           int64_t max_trials_global,
                           ffi::Optional<int64_t> max_trials_per_task = std::nullopt,
                           ffi::Optional<ffi::Array<ffi::String>> op_names = std::nullopt) {
  Target target = Target::Current(false);
  auto pass_func = [=](IRModule m, PassContext ctx) {
    auto max_trials_task = max_trials_per_task.value_or(max_trials_global);
    return MetaScheduleTuner(target, work_dir, max_trials_global, max_trials_task, op_names, params)
        .TuneIRMod(m, ctx);
  };
  return CreateModulePass(/*pass function*/ pass_func, /*opt level*/ 0,
                          /*pass name*/ "MetaScheduleTuneIRModule",
                          /*required*/ {},
                          /*traceable*/ true);
}

Pass MetaScheduleTuneTIR(ffi::String work_dir, int64_t max_trials_global) {
  Target target = Target::Current(false);
  ffi::TypedFunction<tirx::PrimFunc(tirx::PrimFunc, IRModule, PassContext)> pass_func =
      [=](tirx::PrimFunc f, IRModule mod, PassContext ctx) {
        return MetaScheduleTuner(target, work_dir, max_trials_global, max_trials_global,
                                 std::nullopt)
            .TuneTIR(f, ctx);
      };
  return tirx::transform::CreatePrimFuncPass(/*pass function*/ pass_func, /*opt level*/ 0,
                                             /*pass name*/ "MetaScheduleTuneTIR",
                                             /*required*/ {},
                                             /*traceable*/ true);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("relax.transform.MetaScheduleApplyDatabase", MetaScheduleApplyDatabase)
      .def("relax.transform.MetaScheduleTuneIRMod", MetaScheduleTuneIRMod)
      .def("relax.transform.MetaScheduleTuneTIR", MetaScheduleTuneTIR);
}
}  // namespace transform
}  // namespace relax
}  // namespace tvm
