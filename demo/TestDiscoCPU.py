#  這邊是拿simple model 示範跑 disco 單張cpu 的範例 ***
import tvm
from tvm import relax
from tvm.relax.frontend import nn
from tvm import dlight as dl

class SimpleModule(nn.Module):
    def __init__(self):
        super(SimpleModule, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x
    
mod, param_spec = SimpleModule().export_tvm(
    spec={"forward": {"x": nn.spec.Tensor((1, 784), "float32")}}
)
mod = relax.get_pipeline("zero")(mod)
# 以上優化

# build and deployment
import numpy as np
from tvm.runtime import disco



devices = [0]
#disco.ThreadedSession, diso.ProcessSession, diso.SocketSession
#要改成 SocketSession 才能用 mpi
sess = disco.ProcessSession(num_workers=len(devices))
sess.init_ccl("mpi", *devices)






# import os
# path = os.path.join(os.path.dirname(__file__), "test.so")

# dev = tvm.cpu()

# # model inference 可以在這裡編譯，跟TestSimpleModel不一樣的是主要編譯改在這邊呼叫然後沒有export_library
# ex = tvm.compile(mod, target="llvm").export_library(path)
    

# mod = sess.load_vm_module(path)


# # Export to shared library
# lib_path = temp.relpath("test.so")
# executable.export_library(lib_path)
# print(f"Exported library to: {lib_path}")

# # Save parameters separately
# import numpy as np

# params_path = temp.relpath("model_params.npz")
# param_arrays = {f"p_{i}": p.numpy() for i, p in enumerate(params["main"])}
# np.savez(params_path, **param_arrays)
# print(f"Saved parameters to: {params_path}")


# 這邊到時候是rpc過去
#  device_host = "192.168.1.100"  # Replace with your device IP
#  device_port = 9090
#  remote = rpc.connect(device_host, device_port)

#  remote.upload(lib_path)
#  remote.upload(params_path)
#  print("Uploaded files to remote device")

#  # Load the library on the remote device
#  lib = remote.load_module("test.so")

# devies = [0, 1, 2, 3]
#  dev = remote.cpu()

#  # Create VM and load parameters
# for rpc
# vm = relax.VirtualMachine(lib, dev)


#fordisco
# mod = sess.load_vm_module(path)

# #################for rpc
#  # Load parameters from the uploaded file
#  # Note: In practice, you might load this from the remote filesystem
#  params_npz = np.load(params_path)
#  remote_params = [tvm.runtime.tensor(params_npz[f"p_{i}"], dev) for i in range(len(params_npz))]

#  ######################################################################
#  # Step 5: Run Inference on Remote Device
#  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  # Execute the model on the remote ARM device and retrieve results

#  # Prepare input data
#  input_data = np.random.randn(1, 1, 28, 28).astype("float32")
#  remote_input = tvm.runtime.tensor(input_data, dev)
################################


# fordisco
# # 隨機生成
# data_np = np.random.randn(1, 784).astype("float32")
# fc1weight_np = np.random.randn(256, 784).astype("float32")
# fc1bias_np   = np.random.randn(256,).astype("float32")
# fc2weight_np = np.random.randn(10, 256).astype("float32")
# fc2bias_np   = np.random.randn(10,).astype("float32")
# # copy 到 worker 0
# data = sess.copy_to_worker_0(data_np)
# fc1weight = sess.copy_to_worker_0(fc1weight_np)
# fc1bias = sess.copy_to_worker_0(fc1bias_np)
# fc2weight = sess.copy_to_worker_0(fc2weight_np)
# fc2bias = sess.copy_to_worker_0(fc2bias_np)

#  # Run inference on remote device
# forrpc
#  output = vm["main"](remote_input, *remote_params)


#fordisco
# dref_output = mod["forward"](data, fc1weight, fc1bias, fc2weight, fc2bias)
# output_strorage = tvm.runtime.empty((1, 10), "float32", device=dev)
# sess.copy_from_worker_0(output_strorage, dref_output)
# sess.sync_worker_0()
# print("output_strorage:", output_strorage.numpy())


#  # Extract result (handle both tuple and single tensor outputs)
#  if isinstance(output, tvm.ir.Array) and len(output) > 0:
#      result = output[0]
#  else:
#      result = output

#  # Retrieve result from remote device to local
#  result_np = result.numpy()
#  print(f"Inference completed on remote device")
#  print(f"  Output shape: {result_np.shape}")
#  print(f"  Predicted class: {np.argmax(result_np)}")







