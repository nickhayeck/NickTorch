#include "smol_g.h"

void smol_g::fwd() {
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
grad_ball smol_g::bwd(smol_t output) {
	// assert the node from which to backpropagate is an output
	assert(std::find(outputs.begin(), outputs.end(), output.inner) != outputs.end());
	std::queue<smol_t_inner*> queue;
	grad_ball gball(output);
	// append to gball and queue
	gball.append(output, output.get_val().like_ones());
	queue.emplace(output.inner);
	while (!queue.empty()) {
		_bwd(gball, queue);
	}

	return gball;
}

// helper functions for parsing the tree
void smol_g::_trav(smol_t_inner* t) {
	_add(t);
	for (smol_t_inner* up : t->edge_i) {
		_trav(up);
	}
}
void smol_g::_add(smol_t_inner* t) {
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
void smol_g::_fwd(smol_t_inner* t, std::unordered_set<smol_t_inner*>& evaled) {
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
void smol_g::_bwd(grad_ball& gb, std::queue<smol_t_inner*>& q) {
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
