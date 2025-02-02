# Tricks I learned during the implementation

## PyTorch

1. `register_buffer` - A method of `torch.nn`Module` that allows us to register a tensor as a part of the model without it being treated as a trainable parameter. This is useful when storing values like running mean or cos/sin values for RoPE.
 - Has a `persistent` field that controls whether the buffer is included in the models's state dictionary when saving or loading the modelusing `torch.save()` or `torch.load()`.
