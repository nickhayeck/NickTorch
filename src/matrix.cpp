#include "matrix.h"

// minimal floating-point matrix implementation
Matrix Matrix::t() const {
	Matrix out = *this;
	out.is_transposed = (is_transposed) ? false : true;
	return out;
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
Matrix Matrix::operator+(Matrix& b) {
	assert(this->rows == b.rows && this->cols == b.cols);

	Matrix out = *this;

	for(unsigned long i = 0; i < this->val.size(); i++) {
		out.val[i] += b.val[i];
	}

	return out;
}
// sub
Matrix Matrix::operator-(Matrix& b) {
	assert(this->rows == b.rows && this->cols == b.cols);

	Matrix out = *this;

	for(unsigned long i = 0; i < this->val.size(); i++) {
		out.val[i] -= b.val[i];
	}

	return out;
}
// mul
Matrix Matrix::operator*(Matrix& b) {
	assert(this->rows == b.rows && this->cols == b.cols);

	Matrix out = *this;

	for(unsigned long i = 0; i < this->val.size(); i++) {
		out.val[i] *= b.val[i];
	}

	return out;
}
// div
Matrix Matrix::operator/(Matrix& b) {
	assert(this->rows == b.rows && this->cols == b.cols);

	Matrix out = *this;

	for(unsigned long i = 0; i < this->val.size(); i++) {
		out.val[i] /= b.val[i];
	}

	return out;
}
// dot
Matrix Matrix::dot(Matrix& other) {
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