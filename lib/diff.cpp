// includes below
#include <assert.h>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <queue>
// end includes



// minimal floating-point matrix implementation
struct Matrix {
	int rows;
	int cols;
	std::vector<double> val;
	bool is_transposed;
	// scalar init
	Matrix(double d) : rows(1), cols(1), val({d}), is_transposed(false) {}
	//array init
	template<int L>
	Matrix(double (&arr)[L], int N, int K) : 
	rows(N), cols(K), val(arr, arr + L), is_transposed(false) {
		assert(L == N*K);
	}
	// zero init
	Matrix(int N, int K) : rows(N), cols(K), val(N*K), is_transposed(false) {}
	// default constructor
	Matrix() : rows(1), cols(1), val(1), is_transposed(false) {}
	
	Matrix t() const {
		Matrix out = *this;
		out.is_transposed = (is_transposed) ? false : true;
		return out;
	}

	int current_rows() const {
		return (!this->is_transposed) ? this->rows : this->cols;
	}

	int current_cols() const {
		return (!this->is_transposed) ? this->cols : this->rows;
	}

	inline const Matrix like_ones() {
		Matrix out = *this;
		for (double& entry : out.val) {
			entry = 1;
		}

		return out;
	}

	void set(int i, int j, double value) {
		if (!is_transposed) val[i*cols + j] = value;
		else val[j*rows + i] = value;
	}
	double get(int i, int j) {
		return (!is_transposed) ? val[i*cols + j] : val[j*rows + i];
	}
	// matrix-matrix operations
	// add
	Matrix operator+(Matrix& b) {
		assert(this->rows == b.rows && this->cols == b.cols);

		Matrix out = *this;

		for(unsigned long i = 0; i < this->val.size(); i++) {
			out.val[i] += b.val[i];
		}

		return out;
	}

	// sub
	Matrix operator-(Matrix& b) {
		assert(this->rows == b.rows && this->cols == b.cols);

		Matrix out = *this;

		for(unsigned long i = 0; i < this->val.size(); i++) {
			out.val[i] -= b.val[i];
		}

		return out;
	}

	// mul
	Matrix operator*(Matrix& b) {
		assert(this->rows == b.rows && this->cols == b.cols);

		Matrix out = *this;

		for(unsigned long i = 0; i < this->val.size(); i++) {
			out.val[i] *= b.val[i];
		}

		return out;
	}

	// div
	Matrix operator/(Matrix& b) {
		assert(this->rows == b.rows && this->cols == b.cols);

		Matrix out = *this;

		for(unsigned long i = 0; i < this->val.size(); i++) {
			out.val[i] /= b.val[i];
		}

		return out;
	}
	
	// dot
	Matrix dot(Matrix& other) {
		assert(this->current_cols() == other.current_rows());

		Matrix out(this->current_rows(), other.current_cols());
		
		for (int i = 0; i < current_rows(); i++) {
			for (int j = 0; j < other.current_cols(); j++) {
				for (int k = 0; k < current_cols(); k++) {
					out.val[i*current_cols() + j] += this->get(i,k) * other.get(k,j);
				}
			}
		}

		return out;
	}
	
	// exp
	Matrix exp() {
		Matrix out = *this;
		for (unsigned long i = 0; i < val.size(); i++) {
			out.val[i] = std::exp(out.val[i]);
		}

		return out;
	}

};
// utility operations
// print
	void operator<<(std::ostream& o, Matrix& b) {
		for (int i = 0; i < b.current_rows(); i++) {
			for (int j = 0; j < b.current_cols(); j++) {
				o << b.get(i,j) << ",";
			}
			o << std::endl;
		}
	}
// matrix-scalar operations
// mul
Matrix operator*(double d, Matrix b) {
	Matrix out(b);

	for(unsigned long i = 0; i < out.val.size(); i++) {
		out.val[i] *= d;
	}

	return out;
}

// div
Matrix operator/(double d, Matrix b) {
	Matrix out(b);

	for(unsigned long i = 0; i < out.val.size(); i++) {
		out.val[i] = d / out.val[i];
	}

	return out;
}


enum class node_type {
	add,
	sub,
	mul,
	div,
	dot,
	exp,
	input,
	constant,
};

// a wrapper around the internals of smol_t
// so that we can pass around an object and
// not references to that object
struct smol_t_inner {
	// last fwd evaluated value
	Matrix val;

	std::vector<smol_t_inner*> edge_i;
	std::vector<smol_t_inner*> edge_o;

	node_type op;
	// constructor for constant scalar node
	smol_t_inner(double d) : val(Matrix(d)), op(node_type::constant) {}
	// constructor for input node
	smol_t_inner(Matrix& init) :  val(init), edge_i(), edge_o(), op(node_type::input) {}
	// constructor for intermediate node
	template<typename... tensor_t>
	smol_t_inner(node_type o, tensor_t... in) : val(), op(o) {
		edge_i = { in->inner... };
	}

	void eval() {
		switch (op) {
			case node_type::add: {
				// this = lhs + rhs
				assert(edge_i.size() == 2);
				val = edge_i[0]->val + edge_i[1]->val;
				break;
			}

			case node_type::sub: {
				// this = lhs - rhs
				assert(edge_i.size() == 2);
				val = edge_i[0]->val - edge_i[1]->val;
				break;
			}

			case node_type::mul: {
				// this = lhs * rhs
				assert(edge_i.size() == 2);
				val = edge_i[0]->val * edge_i[1]->val;
				break;
			}

			case node_type::div: {
				// this = lhs / rhs
				assert(edge_i.size() == 2);
				val = edge_i[0]->val / edge_i[1]->val;
				break;
			}

			case node_type::dot: {
				// this = lhs @ rhs
				assert(edge_i.size() == 2);
				val = edge_i[0]->val.dot(edge_i[1]->val);
				break;
			}

			case node_type::exp: {
				// this = exp(lhs)
				assert(edge_i.size() == 1);
				val = edge_i[0]->val.exp();
				break;
			}

			default: {
				// no eval for constants or input
			}

		}
	}

	Matrix partial_diff(int parent_index, Matrix acc) const {
		switch(op) {
			case node_type::add: {
				return acc;
			}
			case node_type::sub: {
				// if parent is rhs, negate it
				return (parent_index == 0) ? acc : -1*acc;
			}
			case node_type::mul: {
				// if parent is rhs, return lhs & v.v.
				return (parent_index == 0) ? acc.dot(edge_i[1]->val) : acc.dot(edge_i[0]->val);
			}
			case node_type::div: {
				Matrix div;
				if (parent_index == 0) {
					// lhs
					div = 1 / edge_i[1]->val;
					return acc.dot(div);
				} else if (parent_index == 1) {
					// rhs
					div = -1*edge_i[0]->val / edge_i[1]->val / edge_i[1]->val;
					return acc.dot(div);
				}
			}
			case node_type::dot: {
				// if parent is rhs, return lhs & v.v.
				Matrix tp;
				if (parent_index == 0) {
					tp = edge_i[1]->val.t();
					return acc.dot(tp);
				} else if (parent_index == 1) {
					tp = edge_i[0]->val.t();
					return tp.dot(acc);
				}
			}
			case node_type::exp: {
				Matrix exp = edge_i[0]->val.exp();
				return acc.dot(exp);
			}
			default:{
				return acc;
			}
		}
	}
};

// smol tensor node: a minimal directed acyclic graph.
// this is acyclic because i am asking 
// everyone nicely to not add cycles 👍
struct smol_t {
	smol_t_inner* inner;
	// blanket inner constructor
	template<typename... in_t>
	smol_t(in_t... in) {
		inner = new smol_t_inner(in...);
	}
	node_type get_type() const {
		return inner->op;
	}
	Matrix get_val() const {
		return inner->val;
	}
};


// add
const smol_t operator+(smol_t left, smol_t right) {
	smol_t* out = new smol_t(node_type::add, &left, &right);
	left.inner->edge_o.push_back(out->inner);
	right.inner->edge_o.push_back(out->inner);

	return *out;
}
// sub
const smol_t operator-(smol_t left, smol_t right) {
	smol_t* out = new smol_t(node_type::sub, &left, &right);
	left.inner->edge_o.push_back(out->inner);
	right.inner->edge_o.push_back(out->inner);

	return *out;
}

// mul
const smol_t operator*(smol_t left, smol_t right) {
	smol_t* out = new smol_t(node_type::mul, &left, &right);
	left.inner->edge_o.push_back(out->inner);
	right.inner->edge_o.push_back(out->inner);

	return *out;
}

// div
const smol_t operator/(smol_t left, smol_t right) {
	smol_t* out = new smol_t(node_type::div, &left, &right);
	left.inner->edge_o.push_back(out->inner);
	right.inner->edge_o.push_back(out->inner);

	return *out;
}

// dot
const smol_t dot(smol_t left, smol_t right) {
	smol_t* out = new smol_t(node_type::dot, &left, &right);
	left.inner->edge_o.push_back(out->inner);
	right.inner->edge_o.push_back(out->inner);
	
	return *out;
}

// exp
const smol_t exp(smol_t right) {
	smol_t* out = new smol_t(node_type::exp, &right);
	right.inner->edge_o.push_back(out->inner);
	
	return *out;
}

// gradient ball: holds all of the computed gradients 
// from a backpropagation step and can update the graph 
// TODO: make this an abstract class to allow for various
// optimization methods
struct grad_ball {
	std::unordered_map<smol_t_inner*,Matrix> ball;
	std::unordered_map<smol_t_inner*,int> acc_counter;
public:
	const Matrix get(smol_t t) {
		return get(t.inner);
	}
	const Matrix get(smol_t_inner* t) {
		return ball[t];
	}
	bool contains(smol_t t) {
		return (ball.find(t.inner) != ball.end());
	}
	bool contains(smol_t_inner* t) {
		return (ball.find(t) != ball.end());
	}
	void append(smol_t t, Matrix deriv) {
		ball[t.inner] = deriv;
	}
	void acc(smol_t t, Matrix deriv) {
		acc(t.inner, deriv);
	}
	void acc(smol_t_inner* t, Matrix deriv) {
		if (contains(t)) ball[t] = ball[t] + deriv;
		else ball[t] = deriv;
	}
	// bool update(smol_g& graph) {
	// 	return false;
	// }
};

// smol graph: a wrapper around a collection of smol tensor nodes
struct smol_g {
	// inputs on the graph
	std::vector<smol_t_inner*> inputs;
	// intermediate nodes signifying operations
	std::vector<smol_t_inner*> intermed;
	// outputs on the graph
	std::vector<smol_t_inner*> outputs;
	// constant values
	std::vector<smol_t_inner*> constants;

	// traverse up the graph and add it to the graph model
	template<typename... tensor_t>
	smol_g(tensor_t... out_tensors) {
		outputs = { out_tensors.inner... };

		for (smol_t_inner* output : outputs) {
			for(smol_t_inner* up : output->edge_i) {
				_trav(up);
			}
		}
	}

	void fwd() {
		// place all constants and inputs into the evaluated set
		std::unordered_set<smol_t_inner*> evaled;

		evaled.insert(constants.begin(), constants.end());
		evaled.insert(inputs.begin(), inputs.end());

		for (smol_t_inner* input : inputs) {
			for (smol_t_inner* input_child : input->edge_o) {
				_fwd(input_child, evaled);
			}
		}
	}
	
	// computes the graph's gradient over an output using 
	// breadth-first iteration over the graph starting from
	// the given output node.
	grad_ball bwd(smol_t output) {
		// assert the node from which to backpropagate is an output
		assert(std::find(outputs.begin(), outputs.end(), output.inner) != outputs.end());
		std::queue<smol_t_inner*> queue;
		grad_ball gball;
		// append to gball and queue
		gball.append(output, output.get_val().like_ones());
		queue.emplace(output.inner);
		while (!queue.empty()) {
			_bwd(gball, queue);
		}

		return gball;
	}

private:
	// helper functions for parsing the tree
	void _trav(smol_t_inner* t) {
		_add(t);
		for (smol_t_inner* up : t->edge_i) {
			_trav(up);
		}
	}
	void _add(smol_t_inner* t) {
		if (t->edge_i.size() == 0) {
			// source found! it could be a constant or an input
			if (t->op == node_type::constant && std::find(constants.begin(), constants.end(), t) == constants.end()) {
				constants.push_back(t);
			} else if(std::find(inputs.begin(), inputs.end(), t) == inputs.end()) {
				inputs.push_back(t);
			}
		} else {
			// intermediate found
			if (std::find(intermed.begin(), intermed.end(), t) == intermed.end()) {
				intermed.push_back(t);
			}
		}
	}
	// helper function for forward pass over the graph
	void _fwd(smol_t_inner* t, std::unordered_set<smol_t_inner*>& evaled) {
		if(evaled.find(t) == evaled.end()) {
			bool parents_evaled = true;
			for (smol_t_inner* parent : t->edge_i) {
				if (evaled.find(parent) == evaled.end()) parents_evaled = false;
			}

			if(parents_evaled) {
				// evaluate the node
				t->eval();
				evaled.insert(t);
				for (smol_t_inner* child : t->edge_o) {
					_fwd(child, evaled);
				}
			}
		}
	}
	// helper function for backward pass over the graph
	void _bwd(grad_ball& gb, std::queue<smol_t_inner*>& q) {
		smol_t_inner* t = q.front();
		q.pop();

		Matrix acc_child = gb.get(t);
		for (unsigned long i = 0; i < t->edge_i.size(); i++) {
			smol_t_inner* parent = t->edge_i[i];
			if (!gb.contains(parent)) {
				q.emplace(parent);
			}
			Matrix pd = t->partial_diff(i, acc_child);
			gb.acc(parent, pd);
		}
	}
};

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
	double arr[1] = {2};
	Matrix mat(arr, 1, 1);
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
	// for (auto pair : gb.ball) {
	// 	std::cout << pair.second;
	// 	std::cout << std::endl;
	// }
}
