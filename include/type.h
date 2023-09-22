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
    log,
	input,
	constant,
};

// a wrapper around the internals of nick_t
// so that we can pass around an object and
// not references to that object
struct nick_t_inner {
	// last fwd evaluated value
	Matrix val;

	std::vector<nick_t_inner*> edge_i;
	std::vector<nick_t_inner*> edge_o;

	node_type op;
	// constructor for constant scalar node
	nick_t_inner(double d) : val(Matrix(d)), op(node_type::constant) {}
    // constructor for specifying the form of the underlying matrix
	nick_t_inner(int N, int K) : val(Matrix(N,K)), op(node_type::input) {}
	// constructor for input node
	nick_t_inner(Matrix& init) :  val(init), edge_i(), edge_o(), op(node_type::input) {}
	// constructor for intermediate node
	template<typename... tensor_t>
	nick_t_inner(node_type o, tensor_t... in) : val(), op(o) {
		edge_i = { in->inner... };
	}
	void eval();

	Matrix partial_diff(int parent_index, Matrix acc) const;
};

// tensor node: a minimal directed acyclic graph.
// this is acyclic because i am asking 
// everyone nicely to not add cycles 👍
struct nick_t {
	nick_t_inner* inner;
	// blanket inner constructor
	template<typename... in_t>
	nick_t(in_t... in) {
		inner = new nick_t_inner(in...);
	}
	node_type get_type() const {
		return inner->op;
	}
	Matrix get_val() const {
		return inner->val;
	}
    // some convenient functions for help initializing tensors
    nick_t ones() const {
        return inner->val.ones();
    }
    nick_t eye() const {
        return inner->val.eye();
    }
    nick_t all_scalar(double d) {
        return inner->val.all_scalar(d);
    }
};


// t-t operator definitions 
// add
const nick_t operator+(nick_t left, nick_t right);
// sub
const nick_t operator-(nick_t left, nick_t right);
// mul
const nick_t operator*(nick_t left, nick_t right);
// div
const nick_t operator/(nick_t left, nick_t right);
// dot
const nick_t dot(nick_t left, nick_t right);
// exp
const nick_t exp(nick_t right);
// log
const nick_t log(nick_t right);

// t-s and s-t operator definitions
// add
const nick_t operator+(double left, nick_t right);
const nick_t operator+(nick_t left, double right);
// sub
const nick_t operator-(double left, nick_t right);
const nick_t operator-(nick_t left, double right);
// mul
const nick_t operator*(double left, nick_t right);
const nick_t operator*(nick_t left, double right);
// div
const nick_t operator/(double left, nick_t right);
const nick_t operator/(nick_t left, double right);
