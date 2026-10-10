import psutil
from tvm_ffi import Shape, register_global_func


@register_global_func("runDisco.Memory.usage")
def usage():
    info = psutil.Process().memory_full_info()
    return Shape([info.rss, info.pss])
