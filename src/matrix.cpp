#include "matrix.h"

Matrix Matrix::all_scalar(double d) {
	for (unsigned long i = 0; i < val.size(); i++) val[i] = d;
	return *this;
}


// minimal floating-point matrix implementation
Matrix Matrix::t() const {
	Matrix out = *this;
	out.is_transposed = (is_transposed) ? false : true;
	return out;
}
bool Matrix::is_scalar() const {
	return (rows == 1 && cols == 1);
}

int Matrix::current_rows() const {
	return (!this->is_transposed) ? this->rows : this->cols;
}

int Matrix::current_cols() const {
	return (!this->is_transposed) ? this->cols : this->rows;
}

void Matrix::set(int i, int j, double value) {
	if (!is_transposed) val[i*cols + j] = value;
	else val[j*rows + i] = value;
}
double Matrix::get(int i, int j) {
	return (!is_transposed) ? val[i*cols + j] : val[j*rows + i];
}
// matrix-matrix operations
// add
Matrix operator+(Matrix a, Matrix b) {
	Matrix a_s = broadcast_like(a,b);
	Matrix b_s = broadcast_like(b,a);
	assert(a_s.rows == b_s.rows && a_s.cols == b_s.cols);

	for(unsigned long i = 0; i < a_s.val.size(); i++) {
		a_s.val[i] += b_s.val[i];
	}

	return a_s;
}
// sub
Matrix operator-(Matrix a, Matrix b) {
	Matrix a_s = broadcast_like(a,b);
	Matrix b_s = broadcast_like(b,a);
	assert(a_s.rows == b_s.rows && a_s.cols == b_s.cols);

	for(unsigned long i = 0; i < a_s.val.size(); i++) {
		a_s.val[i] -= a_s.val[i];
	}

	return a_s;
}
// mul
Matrix operator*(Matrix a, Matrix b) {
	Matrix a_s = broadcast_like(a,b);
	Matrix b_s = broadcast_like(b,a);

	assert(a_s.rows == b_s.rows && a_s.cols == b_s.cols);

	for(unsigned long i = 0; i < a_s.val.size(); i++) {
		a_s.val[i] *= b_s.val[i];
	}

	return a_s;
}
// div
Matrix operator/(Matrix a, Matrix b) {
	Matrix a_s = broadcast_like(a,b);
	Matrix b_s = broadcast_like(b,a);
	assert(a_s.rows == b_s.rows && a_s.cols == b_s.cols);

	for(unsigned long i = 0; i < a_s.val.size(); i++) {
		a_s.val[i] /= b_s.val[i];
	}

	return a_s;
}
// dot
Matrix Matrix::dot(Matrix other) {
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
Matrix Matrix::exp() {
	Matrix out = *this;
	for (unsigned long i = 0; i < val.size(); i++) {
		out.val[i] = std::exp(out.val[i]);
	}

	return out;
}


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
// optionally broadcasts `a` to be like `b` if `a` is scalar;
const Matrix broadcast_like(Matrix a, Matrix b) {
	bool a_s = a.is_scalar();
	bool b_s = b.is_scalar();

	if (a_s && b_s) {
		return a;
	} else if (a_s) {
		double scalar = a.get(0,0);
		return Matrix(b.current_rows(), b.current_cols()).all_scalar(scalar);
	} 
	return a;
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