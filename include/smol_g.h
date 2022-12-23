#pragma once

#include "smol_t.h"
#include <unordered_map>
#include <unordered_set>
#include <queue>

// forward declare for use as a friend struct
struct smol_g;
// gradient ball: holds all of the computed gradients 
// from a backpropagation step and can update the graph 
// TODO: make this an abstract class to allow for various
// optimization methods
struct grad_ball {
	friend struct smol_g;
    friend void _bwd(grad_ball& gb, std::queue<smol_t_inner*>& q);
private:
	std::unordered_map<smol_t_inner*,Matrix> ball;
	smol_t output;
	// functions that will be wrapped later for use by the user
    // and are used internally by smol_g 
	bool contains(smol_t_inner* t) {
		return (ball.find(t) != ball.end());
	}
	const Matrix get(smol_t_inner* t) {
		return ball[t];
	}
	void acc(smol_t_inner* t, Matrix deriv) {
		if (contains(t)) ball[t] = ball[t] + deriv;
		else ball[t] = deriv;
	}
public:
    grad_ball(smol_t output) : output(output) {}

	const Matrix get(smol_t t) {
		return get(t.inner);
	}
	bool contains(smol_t t) {
		return (ball.find(t.inner) != ball.end());
	}
	void append(smol_t t, Matrix deriv) {
		ball[t.inner] = deriv;
	}
	void acc(smol_t t, Matrix deriv) {
		acc(t.inner, deriv);
	}

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

	void fwd();
	


private:
	// helper functions for parsing the tree
	void _trav(smol_t_inner* t);
	void _add(smol_t_inner* t);
	// helper function for forward pass over the graph
	void _fwd(smol_t_inner* t, std::unordered_set<smol_t_inner*>& evaled);
};

// computes the graph's gradient over an output using 
// breadth-first iteration over the graph starting from
// the given output node.
grad_ball bwd(smol_t output);
// helper function for backward pass over the graph
void _bwd(grad_ball& gb, std::queue<smol_t_inner*>& q);
// convenience function for doing most of the things we care about at once
// probably not the best thing to use if we don't really care about the 
grad_ball grad(smol_t output);