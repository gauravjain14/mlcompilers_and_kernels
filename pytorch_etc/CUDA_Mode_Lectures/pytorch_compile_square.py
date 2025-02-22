# torch compile
import torch
import logging
from torch.profiler import schedule
from torch.utils._triton import has_triton


device = 'cuda' if torch.cuda.is_available() else 'cpu'

inputs = torch.randn(1000, 1000, device=device)

torch._logging.set_logs(output_code=True)
fn_torch = torch.compile(torch.mul)

with torch.profiler.profile(
    schedule=torch.profiler.schedule(wait=5, warmup=5, active=10, repeat=1)
) as prof:
    for _ in range(20):
        # Should we add the compilation step here and bank on the fact
        # that pytorch will only compile once if the input doesn't change?
        fn_torch(inputs, inputs)
        torch.cuda.synchronize(device)
        prof.step()

    event_list = prof.key_averages()
    print(event_list)