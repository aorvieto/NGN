# NGN

Implementation of the non-negative Gauss-Newton method (NGN) and variants

Simple takeaway: if you are using SGD, instead of doing

```python
 optimizer.step()
```

perform an NGN step!

```python
with torch.no_grad(): 
    grad_norm_squared = get_grad_norm_squared(model)
    lr = sigma/(1+sigma*grad_norm_squared/(2*loss.item()))
    for _, p in enumerate(model.parameters()):
        new_val = p - lr * p.grad
        p.copy_(new_val)
```
