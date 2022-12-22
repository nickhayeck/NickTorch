// includes below
#include "smol_t.h"
#include "smol_g.h"
// end includes

int main() {
	// const int datapts = 1000;
	// tensor x = tensor<100,datapts>(input);
	// tensor y = tensor<10,datapts>(output);
	
	// const int l1 = 10;
	// tensor w1 = tensor<100,l1>().rand(); 
	// tensor b1 = tensor<l1,1>().rand(); 

	// const int l2 = 10;
	// tensor w2 = tensor<l1,l2>().rand(); 
	// tensor b2 = tensor<l2,1>().rand(); 

	// // Two-layer NN
	// tensor s1 = 1 / (1 - exp(-1 * dot(x, w1) - b1));
	// tensor s2 = 1 / (1 - exp(-1 * dot(s2, w2) - b2));
	// // MSE
	// tensor loss = dot(s2 - s1, s2 - s1) 

	// optimizer opt(loss);
	double arr[4] = {1,1,1,1};
	Matrix mat(arr, 2, 2);
	smol_t st1(mat);
	smol_t st2(mat);
	smol_t st3(mat);

	smol_t out1 = exp(st1*st2)*st1/st2 + st1*st2;
	// smol_t out2 = st2 * st3 + dot(st1, st2);
	smol_g graph(out1);

	std::cout << "Types connected to `out1`: ";
	for (unsigned long i = 0; i < out1.inner->edge_i.size(); i++) std::cout << (int) out1.inner->edge_i[i]->op << ", ";
	std::cout << std::endl;


	std::cout << "# Input Nodes: " << graph.inputs.size() << std::endl;
	std::cout << "# Inter Nodes: " << graph.intermed.size() << std::endl;
	std::cout << "# Outpt Nodes: " << graph.outputs.size() << std::endl;
	std::cout << "# Const Nodes: " << graph.constants.size() << std::endl;

	graph.fwd();

	std::cout << "Fwd Outputs: ";
	for (unsigned long i = 0; i < graph.outputs.size(); i++) std::cout << graph.outputs[i]->val.get(0,0) << ",";
	std::cout << std::endl;

	grad_ball gb = graph.bwd(out1);

	std::cout << "del(out1) / del(st1)" << std::endl;
	Matrix grad = gb.get(st1);
	std::cout << grad;
}
