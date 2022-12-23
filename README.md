# `smol-grad`: a minimal autodiff implementation in C++
Have you ever wanted derivatives but you're stuck with plain, old floating point operations? Have you ever wanted your operations to construct a computational graph in the background? Meet `smol-grad`. With this library you can turn boring old C++ like `double z = x * y + log(x) + exp(x * y) / y` into cool, sick, and differentiable code just by changing a data type: `smol_t z = x * y + log(x) + exp(x * y) / y`.

Now all you have to do is grab its gradient like so: `grad(z)`, and you have every gradient one could desire (as long as you are desiring the gradients of `x` or `y`, that is).

Being able to take gradients on a string of operations automatically is the basis for easy and efficient creation of machine learning models. If the folks over at OpenAI had to hand-compute the gradients for every model they built and program that in, [ChatGPT](https://openai.com/blog/chatgpt/) may have taken a few more centuries to come out!


## TODO:
- [x] Split into multiple files, use cmake to compile
- [x] Break out into headers and impl files
- [x] Change `partialdiff` to be a non-member function
- [x] create `smol_t` and scalar operations (both directions)
- [x] create `smol_t` unary operation `log`
- [x] round out `Matrix` and scalar operations (both directions), also add negation operator.
- [x] phase out `Matrix` current_rows and current_cols functions, instead just swap rows and columns variables
- [x] update `smol_t` creation interface, eliminating any reference to constant matricies. Should appear like `smol_t(xdim,ydim).ones()` or `smol_t(xdim,ydim).scalar(10)` or `smol_t(xdim,ydim).eye()`
- [x] update the matrix constructors to match
- [x] have the `grad_ball` contain the output with respect to which the gradient was taken
- [x] convenience function `grad_ball grad(smol_t output)` so that syntax is: `smol_t z = x + y; grad_ball gb = grad(z)`
- [x] better example in main
- [ ] more tests!
- [ ] go on a commenting spree 
- [ ] better README
- [ ] better destructors for the graph. we are def leaking memory atm.