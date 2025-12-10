# nccl-skew-analyzer

nsys-ui makes it difficult to analyze when ranks in a NCCL collective fall out of sync, which can be very harmful to performance. This utility tries to match collective kernel launches up and measure differences in launch time and duration to help understand how your application is being affected. 

### Usage

Use nsys-ui to export the .nsys-rep file as a sqlite file (File > Export). Then run `cargo run -- [sqlite file]`. 

### Example output

```
NCCL kernels present on all ranks (skew based on kernel start):

ncclDevKernel_AllReduce_Sum_bf16_RING_LL:
  occurrence #0: skew 5.264ms (8 ranks)
    pid 549137 (mixlayer-modeld): +5.264ms from earliest, start=29860643859 ns, duration=46.432us (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549142 (mixlayer-modeld): +2.619ms from earliest, start=29857999294 ns, duration=2.691ms (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549148 (mixlayer-modeld): +3.727ms from earliest, start=29859106621 ns, duration=1.584ms (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549154 (mixlayer-modeld): +3.846ms from earliest, start=29859225620 ns, duration=1.465ms (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549160 (mixlayer-modeld): +3.818ms from earliest, start=29859198053 ns, duration=1.492ms (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549166 (mixlayer-modeld): +4.096ms from earliest, start=29859475899 ns, duration=1.215ms (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549172 (mixlayer-modeld): +0.000ns from earliest, start=29855379950 ns, duration=5.310ms (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549179 (mixlayer-modeld): +4.828ms from earliest, start=29860208152 ns, duration=482.653us (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
  occurrence #1: skew 13.484ms (8 ranks)
    pid 549137 (mixlayer-modeld): +13.484ms from earliest, start=29878266408 ns, duration=40.768us (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549142 (mixlayer-modeld): +2.583ms from earliest, start=29867366158 ns, duration=10.941ms (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549148 (mixlayer-modeld): +0.000ns from earliest, start=29864782831 ns, duration=13.524ms (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549154 (mixlayer-modeld): +1.301ms from earliest, start=29866084296 ns, duration=12.222ms (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549160 (mixlayer-modeld): +1.034ms from earliest, start=29865816923 ns, duration=12.490ms (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549166 (mixlayer-modeld): +12.041ms from earliest, start=29876823618 ns, duration=1.483ms (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549172 (mixlayer-modeld): +4.147ms from earliest, start=29868930043 ns, duration=9.377ms (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549179 (mixlayer-modeld): +12.258ms from earliest, start=29877040705 ns, duration=1.266ms (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
  occurrence #2: skew 817.053us (8 ranks)
    pid 549137 (mixlayer-modeld): +817.053us from earliest, start=29879251517 ns, duration=40.895us (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549142 (mixlayer-modeld): +2.506us from earliest, start=29878436970 ns, duration=855.414us (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549148 (mixlayer-modeld): +0.000ns from earliest, start=29878434464 ns, duration=857.721us (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549154 (mixlayer-modeld): +2.048us from earliest, start=29878436512 ns, duration=855.863us (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549160 (mixlayer-modeld): +333.000ns from earliest, start=29878434797 ns, duration=857.175us (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549166 (mixlayer-modeld): +4.148us from earliest, start=29878438612 ns, duration=853.816us (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549172 (mixlayer-modeld): +1.962us from earliest, start=29878436426 ns, duration=855.993us (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
    pid 549179 (mixlayer-modeld): +3.191us from earliest, start=29878437655 ns, duration=854.554us (ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<(unsigned long)4096>))
```