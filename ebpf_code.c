#include <uapi/linux/ptrace.h>
#include <linux/sched.h>

#define MAX_SYSCALLS 512
#define TARGET_PID 0

BPF_PERCPU_ARRAY(histogram, u32, MAX_SYSCALLS);

TRACEPOINT_PROBE(raw_syscalls, sys_enter)
{
    u64 pid = bpf_get_current_pid_tgid() >> 32;
    if(pid != TARGET_PID) {
        return 0;
    }
    
    u32 key = (u32)args->id;
    u32 value = 0, *pval = NULL;
    pval = histogram.lookup_or_try_init(&key, &value);
    if(pval) {
        *pval += 1;
    }
    return 0;
}
