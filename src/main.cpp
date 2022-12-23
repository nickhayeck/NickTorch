// includes below
#include "smol_t.h"
#include "smol_g.h"
// end includes

int main() {
	smol_t x = smol_t(2);
	smol_t y = smol_t(3);

	smol_t z = x * y + log(x) + exp(x * y) / y;

	grad_ball gb = grad(z);

	std::cout << "dz/dx: ";
	Matrix grad_x = gb.get(x);
	std::cout << grad_x;

	std::cout << "dz/dy: ";
	Matrix grad_y = gb.get(y);
	std::cout << grad_y;
}
