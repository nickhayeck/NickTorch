// includes below
#include "type.h"
#include "graph.h"
// end includes

int main() {
	nick_t x = nick_t(2);
	nick_t y = nick_t(3);

	nick_t z = x * y + log(x) + exp(x * y) / y;

	grad_ball gb = grad(z);

	std::cout << "dz/dx: ";
	Matrix grad_x = gb.get(x);
	std::cout << grad_x;

	std::cout << "dz/dy: ";
	Matrix grad_y = gb.get(y);
	std::cout << grad_y;
}
