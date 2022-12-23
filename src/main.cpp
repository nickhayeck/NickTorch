// includes below
#include "smol_t.h"
#include "smol_g.h"
// end includes

int main() {
	smol_t X = smol_t(3,5).all_scalar(3);

	smol_t w1 = smol_t(3,5).all_scalar(3);
	smol_t b1 = smol_t(3,5).all_scalar(1);

	smol_t out1 = w1 * X + b1;
	// smol_t loss = 
	
	smol_g graph(out1);

	graph.fwd();

	std::cout << "Fwd Outputs: ";
	for (unsigned long i = 0; i < graph.outputs.size(); i++) std::cout << graph.outputs[i]->val.get(0,0) << ",";
	std::cout << std::endl;

	grad_ball gb = graph.bwd(out1);

	std::cout << "del(out1) / del(w1)" << std::endl;
	Matrix grad = gb.get(w1);
	std::cout << grad;
}
