#include "smol_t.h"

// a wrapper around the internals of smol_t
// so that we can pass around an object and
// not references to that object

void smol_t_inner::eval() {
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

Matrix smol_t_inner::partial_diff(int parent_index, Matrix acc) const {
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
			return (parent_index == 0) ? acc * edge_i[1]->val : acc * edge_i[0]->val;
		}
		case node_type::div: {
			Matrix div;
			if (parent_index == 0) {
				// lhs
				div = 1 / edge_i[1]->val;
				return acc * div;
			} else if (parent_index == 1) {
				// rhs
				div = -1*edge_i[0]->val / edge_i[1]->val / edge_i[1]->val;
				return acc * div;
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
			return acc * exp;
		}
		default:{
			return acc;
		}
	}
}


// t-t operator definitions 
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
