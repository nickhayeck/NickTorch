#pragma once
#include "matrix.h"
#include <vector>


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
	void eval();

	Matrix partial_diff(int parent_index, Matrix acc) const;
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


// t-t operator definitions 
// add
const smol_t operator+(smol_t left, smol_t right);
// sub
const smol_t operator-(smol_t left, smol_t right);
// mul
const smol_t operator*(smol_t left, smol_t right);
// div
const smol_t operator/(smol_t left, smol_t right);
// dot
const smol_t dot(smol_t left, smol_t right);
// exp
const smol_t exp(smol_t right);