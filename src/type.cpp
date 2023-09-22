#include "type.h"

// a wrapper around the internals of nick_t
// so that we can pass around an object and
// not references to that object

void nick_t_inner::eval() {
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
			// this = exp(rhs)
			assert(edge_i.size() == 1);
			val = edge_i[0]->val.exp();
			break;
		}

		case node_type::log: {
			// this = log(rhs)
			assert(edge_i.size() == 1);
			val = edge_i[0]->val.log();
			break;
		}
		case node_type::input:
		case node_type::constant: {
			break;
		}
		default: {
			std::cout << "Forward pass does not implement this operation!" << std::endl;
			std::exit(1);
		}

	}
}

Matrix partial_diff(nick_t t, int parent_index, Matrix acc) {
	return t.inner->partial_diff(parent_index, acc);
}

Matrix nick_t_inner::partial_diff(int parent_index, Matrix acc) const {
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
			if (parent_index == 0) {
				Matrix a = broadcast_like(acc,edge_i[1]->val);
				Matrix rhs = broadcast_like(edge_i[1]->val,acc);

				return a * rhs;
			} else {
				Matrix a = broadcast_like(acc,edge_i[0]->val);
				Matrix lhs = broadcast_like(edge_i[0]->val,acc);

				return a * lhs;
			}
		}
		case node_type::div: {
			Matrix div;
			if (parent_index == 0) {
				// lhs
				
				div = 1 / edge_i[1]->val;
				Matrix a = broadcast_like(acc, div);
				Matrix d = broadcast_like(div, acc);
				return a * d;
			} else if (parent_index == 1) {
				// rhs
				div = -edge_i[0]->val / edge_i[1]->val / edge_i[1]->val;
				Matrix a = broadcast_like(acc, div);
				Matrix d = broadcast_like(div, acc);
				return a * d;
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
		case node_type::log: {
			Matrix log = 1 / edge_i[0]->val;
			return acc * log;
		}
		case node_type::input:
		case node_type::constant: {
			return acc;
		}
		default:{
			std::cout << "Backward pass does not implement this operation!" << std::endl;
			std::exit(1);
		}
	}
}


// t-t operator definitions 
// add
const nick_t operator+(nick_t left, nick_t right) {
	nick_t* out = new nick_t(node_type::add, &left, &right);
	left.inner->edge_o.push_back(out->inner);
	right.inner->edge_o.push_back(out->inner);

	return *out;
}
// sub
const nick_t operator-(nick_t left, nick_t right) {
	nick_t* out = new nick_t(node_type::sub, &left, &right);
	left.inner->edge_o.push_back(out->inner);
	right.inner->edge_o.push_back(out->inner);

	return *out;
}

// mul
const nick_t operator*(nick_t left, nick_t right) {
	nick_t* out = new nick_t(node_type::mul, &left, &right);
	left.inner->edge_o.push_back(out->inner);
	right.inner->edge_o.push_back(out->inner);

	return *out;
}

// div
const nick_t operator/(nick_t left, nick_t right) {
	nick_t* out = new nick_t(node_type::div, &left, &right);
	left.inner->edge_o.push_back(out->inner);
	right.inner->edge_o.push_back(out->inner);

	return *out;
}

// dot
const nick_t dot(nick_t left, nick_t right) {
	nick_t* out = new nick_t(node_type::dot, &left, &right);
	left.inner->edge_o.push_back(out->inner);
	right.inner->edge_o.push_back(out->inner);
	
	return *out;
}

// exp
const nick_t exp(nick_t right) {
	nick_t* out = new nick_t(node_type::exp, &right);
	right.inner->edge_o.push_back(out->inner);
	
	return *out;
}

// log
const nick_t log(nick_t right) {
	nick_t* out = new nick_t(node_type::log, &right);
	right.inner->edge_o.push_back(out->inner);
	
	return *out;
}

// t-s and s-t operator definitions
// add
const nick_t operator+(double left, nick_t right) {
	nick_t* cons = new nick_t(left);
	nick_t* out = new nick_t(node_type::add, cons, &right);
	cons->inner->edge_o.push_back(out->inner);
	right.inner->edge_o.push_back(out->inner);

	return *out;
}
const nick_t operator+(nick_t left, double right) {
	nick_t* cons = new nick_t(right);
	nick_t* out = new nick_t(node_type::add, &left, cons);
	cons->inner->edge_o.push_back(out->inner);
	left.inner->edge_o.push_back(out->inner);

	return *out;
}
// sub
const nick_t operator-(double left, nick_t right) {
	nick_t* cons = new nick_t(left);
	nick_t* out = new nick_t(node_type::sub, cons, &right);
	cons->inner->edge_o.push_back(out->inner);
	right.inner->edge_o.push_back(out->inner);

	return *out;
}
const nick_t operator-(nick_t left, double right) {
	nick_t* cons = new nick_t(right);
	nick_t* out = new nick_t(node_type::sub, &left, cons);
	cons->inner->edge_o.push_back(out->inner);
	left.inner->edge_o.push_back(out->inner);

	return *out;
}
// mul
const nick_t operator*(double left, nick_t right) {
	nick_t* cons = new nick_t(left);
	nick_t* out = new nick_t(node_type::mul, cons, &right);
	cons->inner->edge_o.push_back(out->inner);
	right.inner->edge_o.push_back(out->inner);

	return *out;
}
const nick_t operator*(nick_t left, double right) {
	nick_t* cons = new nick_t(right);
	nick_t* out = new nick_t(node_type::mul, &left, cons);
	cons->inner->edge_o.push_back(out->inner);
	left.inner->edge_o.push_back(out->inner);

	return *out;
}
// div
const nick_t operator/(double left, nick_t right) {
	nick_t* cons = new nick_t(left);
	nick_t* out = new nick_t(node_type::div, cons, &right);
	cons->inner->edge_o.push_back(out->inner);
	right.inner->edge_o.push_back(out->inner);

	return *out;
}
const nick_t operator/(nick_t left, double right) {
	nick_t* cons = new nick_t(right);
	nick_t* out = new nick_t(node_type::div, &left, cons);
	cons->inner->edge_o.push_back(out->inner);
	left.inner->edge_o.push_back(out->inner);

	return *out;
}

