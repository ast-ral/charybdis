use std::ops::{Index, IndexMut};

pub const NUM_DIMS: usize = 3;

pub struct Tensor<T: TensorIndex> {
	backing_storage: T::BackingStorage,
}

pub struct Index1([usize; 1]);
pub struct Index2([usize; 2]);
pub struct Index3([usize; 3]);

pub trait TensorIndex {
	type BackingStorage;

	fn new_backing_storage() -> Self::BackingStorage;
	fn index<'a>(&self, backing: &'a Self::BackingStorage) -> &'a f64;
	fn index_mut<'a>(&self, backing: &'a mut Self::BackingStorage) -> &'a mut f64;
}

impl TensorIndex for Index1 {
	type BackingStorage = [f64; NUM_DIMS];

	fn new_backing_storage() -> [f64; NUM_DIMS] {
		[0.0; NUM_DIMS]
	}

	fn index<'a>(&self, backing: &'a [f64; NUM_DIMS]) -> &'a f64 {
		debug_assert!(self.0[0] < NUM_DIMS);

		&backing[self.0[0]]
	}

	fn index_mut<'a>(&self, backing: &'a mut [f64; NUM_DIMS]) -> &'a mut f64 {
		debug_assert!(self.0[0] < NUM_DIMS);

		&mut backing[self.0[0]]
	}
}

impl TensorIndex for Index2 {
	type BackingStorage = [f64; NUM_DIMS * NUM_DIMS];

	fn new_backing_storage() -> [f64; NUM_DIMS * NUM_DIMS] {
		[0.0; NUM_DIMS * NUM_DIMS]
	}

	fn index<'a>(&self, backing: &'a [f64; NUM_DIMS * NUM_DIMS]) -> &'a f64 {
		debug_assert!(self.0[0] < NUM_DIMS);
		debug_assert!(self.0[1] < NUM_DIMS);

		&backing[self.0[0] * NUM_DIMS + self.0[1]]
	}

	fn index_mut<'a>(&self, backing: &'a mut [f64; NUM_DIMS * NUM_DIMS]) -> &'a mut f64 {
		debug_assert!(self.0[0] < NUM_DIMS);
		debug_assert!(self.0[1] < NUM_DIMS);

		&mut backing[self.0[0] * NUM_DIMS + self.0[1]]
	}
}

impl TensorIndex for Index3 {
	type BackingStorage = [f64; NUM_DIMS * NUM_DIMS * NUM_DIMS];

	fn new_backing_storage() -> [f64; NUM_DIMS * NUM_DIMS * NUM_DIMS] {
		[0.0; NUM_DIMS * NUM_DIMS * NUM_DIMS]
	}

	fn index<'a>(&self, backing: &'a [f64; NUM_DIMS * NUM_DIMS * NUM_DIMS]) -> &'a f64 {
		debug_assert!(self.0[0] < NUM_DIMS);
		debug_assert!(self.0[1] < NUM_DIMS);
		debug_assert!(self.0[2] < NUM_DIMS);

		&backing[self.0[0] * NUM_DIMS * NUM_DIMS + self.0[1] * NUM_DIMS + self.0[2]]
	}

	fn index_mut<'a>(&self, backing: &'a mut [f64; NUM_DIMS * NUM_DIMS * NUM_DIMS]) -> &'a mut f64 {
		debug_assert!(self.0[0] < NUM_DIMS);
		debug_assert!(self.0[1] < NUM_DIMS);
		debug_assert!(self.0[2] < NUM_DIMS);

		&mut backing[self.0[0] * NUM_DIMS * NUM_DIMS + self.0[1] * NUM_DIMS + self.0[2]]
	}
}

impl From<[usize; 1]> for Index1 {
	fn from(array: [usize; 1]) -> Self {
		Index1(array)
	}
}

impl From<[usize; 2]> for Index2 {
	fn from(array: [usize; 2]) -> Self {
		Index2(array)
	}
}

impl From<[usize; 3]> for Index3 {
	fn from(array: [usize; 3]) -> Self {
		Index3(array)
	}
}

impl<T: TensorIndex> Tensor<T> {
	pub fn new() -> Self {
		Tensor {
			backing_storage: T::new_backing_storage(),
		}
	}
}

impl<T: TensorIndex, C: Into<T>> Index<C> for Tensor<T> {
	type Output = f64;

	fn index(&self, indexes: C) -> &Self::Output {
		indexes.into().index(&self.backing_storage)
	}
}

impl<T: TensorIndex, C: Into<T>> IndexMut<C> for Tensor<T> {
	fn index_mut(&mut self, indexes: C) -> &mut Self::Output {
		indexes.into().index_mut(&mut self.backing_storage)
	}
}
