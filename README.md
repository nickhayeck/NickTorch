# `smol-grad`: a minimal autodiff implementation in C++


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
- [ ] better example in main
- [ ] more tests!
- [ ] go on a commenting spree 
- [ ] better README
- [ ] convenience function `grad_ball grad(smol_t output)` so that syntax is: `smol_t z = x + y; grad_ball gb = grad(z)`
- [ ] better destructors for the graph. we are def leaking memory atm.