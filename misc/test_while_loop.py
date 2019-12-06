from jax import lax



def body_fun(val):
    u[0]+=1
    return val+1

cond_fun = lambda j: j<10


### python
u=[0]
val=0
def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val

val=while_loop(cond_fun, body_fun, val)
print('python', u, val)

### xla
u=[0]
val=0
val=lax.while_loop(cond_fun, body_fun, val)
print('xla', u,val)
    

